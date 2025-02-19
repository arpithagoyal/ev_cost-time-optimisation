import traci
import numpy as np
from scipy.spatial.distance import cdist
import timeit
import random
import traceback
import xml.etree.ElementTree as ET
import re

from sklearn.preprocessing import StandardScaler

class Mysimulation:
    
    def __init__(self, all_agents, TrafficGen, sumo_cmd, max_steps, num_states, num_actions, training_epochs):
        self._TrafficGen = TrafficGen
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._step = 0
        self._num_states = num_states
        self._num_actions = num_actions
        self._all_agents = all_agents
        self._training_epochs = training_epochs
        self._charging_stations=[]
        self._standard_speed= 14 #m/s
        self._standard_power_consumption_rate=10 #kW
        self._standard_charging_cost=10 #rupees/kWh
        #initialise time_matrix
        self._time_matrix={}
        #initialise cost matrix
        self._cost_matrix={}
        #initialise waiting times for each cs 
        self.waiting_times = {}


    def initialize_time_and_cost_matrix(self):
        for cs_i in self._charging_stations:
            self._time_matrix[cs_i] = {}
            self._cost_matrix[cs_i]={}
            for cs_j in self._charging_stations:
                self._time_matrix[cs_i][cs_j] = 0
                self._cost_matrix[cs_i][cs_j] = 0

    def get_distance(self, cs_i, cs_j):
        pos_i = traci.chargingstation.getPosition(cs_i)
        pos_j = traci.chargingstation.getPosition(cs_j)
        return traci.simulation.getDistance2D(pos_i[0], pos_i[1], pos_j[0], pos_j[1], isDriving=True)

    def update_time_cost_matrix(self):
        for cs_i in self._charging_stations:
            for cs_j in self._charging_stations:
                if(cs_i!=cs_j):
                    distance=self.get_distance(cs_i, cs_j)
                    time=distance/self._standard_speed
                    cost=self._standard_power_consumption_rate*(time/3600)*self._standard_charging_cost
                    self._time_matrix[cs_i][cs_j]=time*self.get_traffic_factor(cs_i,cs_j)
                    self._cost_matrix[cs_i][cs_j]=cost

    def get_traffic_factor(self, cs_i, cs_j):
        edge_i = traci.chargingstation.getLaneID(cs_i)
        edge_j = traci.chargingstation.getLaneID(cs_j)
        
        # Get the actual edge IDs
        edge_i = traci.lane.getEdgeID(edge_i)
        edge_j = traci.lane.getEdgeID(edge_j)
        
        route = traci.simulation.findRoute(edge_i, edge_j)
        total_time = sum(traci.edge.getTraveltime(edge) for edge in route.edges)
        free_flow_time = sum(traci.edge.getLength(edge) / traci.edge.getMaxSpeed(edge) for edge in route.edges)
        
        # Ensure the factor is always >= 1
        return max(1, total_time / free_flow_time)
    
    def update_waiting_times(self):
        for cs_id in self._charging_stations:
            total_waiting_time = 0
            vehicles = traci.chargingstation.getVehicleIDs(cs_id)
            
            for vehicle in vehicles:
                charging_time = self.calculate_charging_time(cs_id, vehicle)
                total_waiting_time += charging_time
            
            self.waiting_times[cs_id] = total_waiting_time


    def calculate_charging_time(self, cs_id, vehicle):
        # Get charging station power
        power = float(traci.chargingstation.getParameter(cs_id, "power"))
        
        # Get vehicle's battery capacity and current charge
        battery_capacity = float(traci.vehicle.getParameter(vehicle, "maximumBatteryCapacity"))
        current_charge = float(traci.vehicle.getParameter(vehicle, "actualBatteryCapacity"))
        
        # Calculate charging time
        charge_needed = battery_capacity - current_charge
        charging_time = (charge_needed / power) * 3600  # Convert to seconds
        
        return charging_time

    def run(self,episode):
        # #Generate charging stations file for this simulation
        self._TrafficGen.generate_chargingstationfile(seed=episode)

        # first, generate the route file for this simulation and set up sumo
        traci.start(self._sumo_cmd)
        print("Simulating...")
        start_time = timeit.default_timer()
        #get all the charging stations
        self._charging_stations = traci.chargingstation.getIDList()
        #update time and cost matrix
        self.initialize_time_and_cost_matrix()
        self.update_time_cost_matrix()
        #update waiting time for each cs
        self.update_waiting_times()
        

        self._step = 0
        x=10
        while (self._step<self._max_steps):
            #get the current state vector of 65
            current_state,vids,csids,nearest_cs_ids=self.get_current_state()
            actions=[]
            #decay epsilon
            if(self._step%x==0):
                for agent in self._all_agents:
                    agent.decay_epsilon()
            # predict action from each model and append to action array 
            for agent in self._all_agents:
                actions.append(agent.greedy_actor(current_state))
            #simulate all actions
            assignments=self.assign_vehicles_to_charging_stations(vids,csids,actions)
            self.route_vehicles_to_charging_stations(assignments)
            #calculate reward
            reward=self.calculate_reward(vids,nearest_cs_ids,assignments)
            #get next state
            next_state=self.get_current_state()
            # put in memory
            for agent in self._all_agents:
                agent.observe((current_state, actions, reward, next_state))
            # call replay function to train model 
            print("Training...")
            start_time = timeit.default_timer()
            for _ in range(self._training_epochs):
                print("training epoch {}".format(_))
                for agent in self._all_agents:
                    agent.replay()
            training_time = round(timeit.default_timer() - start_time, 1)
            print("Training time : ", training_time)
            #update target model every episode
            for agent in self._all_agents:
                agent.update_target_model()
            self._step=self._step+1
        simulation_time = round(timeit.default_timer() - start_time, 1)
        return simulation_time,0

    def get_distance_vehicle_to_cs(self, vehicle_id, cs_id):
        vehicle_pos = traci.vehicle.getPosition(vehicle_id)
        cs_pos = traci.chargingstation.getPosition(cs_id)
        return traci.simulation.getDistance2D(vehicle_pos[0], vehicle_pos[1], cs_pos[0], cs_pos[1], isDriving=True)
    
    def calculate_charging_cost(self, vehicle_id, assigned_cs_id):
        current_charge = float(traci.vehicle.getParameter(vehicle_id, "actualBatteryCapacity"))
        max_battery_capacity = float(traci.vehicle.getParameter(vehicle_id, "maximumBatteryCapacity"))
        charge_needed = max_battery_capacity - current_charge
        
        # Get the charging cost for the specific charging station
        cs_charging_cost = float(traci.chargingstation.getParameter(assigned_cs_id, "cost"))
        
        return charge_needed * cs_charging_cost

    
    def calculate_reward(self, vids, nearest_cs_ids, assignments):
        times = []
        costs = []

        for i, vid in enumerate(vids):
            assigned_cs = assignments.get(vid)
            if assigned_cs is None:
                print("error:no assigned cs for this vehicle",vid)
                continue  # Skip if no assignment for this vehicle

            nearest_cs = nearest_cs_ids[i]

            # Calculate time and cost to reach nearest CS
            distance_to_nearest = self.get_distance_vehicle_to_cs(vid, nearest_cs)
            time_to_nearest = distance_to_nearest / self._standard_speed
            cost_to_nearest = (self._standard_power_consumption_rate * 
                            (time_to_nearest / 3600) * 
                            self._standard_charging_cost)

            # Calculate total time
            time = (time_to_nearest + 
                    self._time_matrix[nearest_cs][assigned_cs] + 
                    self.waiting_times[assigned_cs])

            # Calculate total cost
            cost = (cost_to_nearest + 
                    self._cost_matrix[nearest_cs][assigned_cs] + 
                    self.calculate_charging_cost(vid))

            times.append(time)
            costs.append(cost)

        # Normalize times and costs
        max_time = max(times) if times else 1
        max_cost = max(costs) if costs else 1
        normalized_times = [t / max_time for t in times]
        normalized_costs = [c / max_cost for c in costs]

        # Calculate reward
        rewards = [0.5 * (1 - nt) + 0.5 * (1 - nc) for nt, nc in zip(normalized_times, normalized_costs)]
        total_reward = sum(rewards)

        return total_reward
    
    def route_vehicles_to_charging_stations(self, assignments):
        for vehicle_id, charging_station_id in assignments.items():
            try:
                # Get the current edge of the vehicle
                current_edge = traci.vehicle.getRoadID(vehicle_id)
                
                # Get the edge of the charging station
                cs_lane = traci.chargingstation.getLaneID(charging_station_id)
                cs_edge = traci.lane.getEdgeID(cs_lane)
                
                # Find a route from the current edge to the charging station edge
                route = traci.simulation.findRoute(current_edge, cs_edge)
                
                # Set the new route for the vehicle
                traci.vehicle.setRoute(vehicle_id, route.edges)
                
                print(f"Routed vehicle {vehicle_id} to charging station {charging_station_id}")
            except traci.TraCIException as e:
                print(f"Error routing vehicle {vehicle_id}: {str(e)}")
        
        # Move the simulation one step forward after updating all routes
        traci.simulationStep()

    def assign_vehicles_to_charging_stations(self, vids, csids, actions):
        assignments = {}
        for i, vehicle_id in enumerate(vids):
            if i < len(actions):  # Ensure we have an action for this vehicle
                action = actions[i]
                if 0 <= action < 5:  # Validate action is in range 0-4
                    charging_stations = csids[vehicle_id]
                    if action < len(charging_stations):
                        assigned_cs = charging_stations[action]
                        assignments[vehicle_id] = assigned_cs
                    else:
                        print(f"Warning: Action {action} out of range for vehicle {vehicle_id}")
                else:
                    print(f"Warning: Invalid action {action} for vehicle {vehicle_id}")
        return assignments
    def get_current_state(self):
        # Get all vehicles with charge less than 50%
        low_charge_vehicles = [v for v in traci.vehicle.getIDList() 
                               if float(traci.vehicle.getParameter(v, "actualBatteryCapacity")) / 
                                  float(traci.vehicle.getParameter(v, "maximumBatteryCapacity")) < 0.5]

        # Sort by charge level and get up to 5 lowest
        low_charge_vehicles.sort(key=lambda v: float(traci.vehicle.getParameter(v, "actualBatteryCapacity")))
        target_vehicles = low_charge_vehicles[:5]
        state_vector = []
        csids={} 
        nearest_cs_ids=[]
        for v in target_vehicles:
            csids[v]=[]
        for i in range(5):
            if i < len(target_vehicles):
                vehicle = target_vehicles[i]
                v_charge = float(traci.vehicle.getParameter(vehicle, "actualBatteryCapacity"))
                v_lane = traci.vehicle.getLaneID(vehicle)
                v_pos = traci.vehicle.getLanePosition(vehicle)

                nearest_cs = self.find_nearest_cs(v_lane, v_pos)
                nearest_cs_ids.append(nearest_cs)
                nearest_5_cs = self.find_nearest_5_cs(nearest_cs)
                v_distance = self.calculate_distance(v_lane, v_pos, nearest_cs)

                state_vector.extend([v_charge, v_lane, v_distance])

                for cs in nearest_5_cs:
                    csids[vehicle].append(cs)
                    cs_cost = float(traci.chargingstation.getParameter(cs, "cost"))
                    cs_power = float(traci.chargingstation.getParameter(cs, "power"))
                    state_vector.extend([cs_cost, cs_power])
            else:
                # Add dummy vehicle with 100% charge and all other values as 0
                state_vector.extend([100, 0, 0])  # 100% charge, dummy lane, 0 distance
                state_vector.extend([0, 0] * 5)  # 5 dummy charging stations with 0 cost and 0 power

        return np.array(state_vector), target_vehicles, csids,nearest_cs_ids

    def find_nearest_cs(self, lane, pos):
        lane_cs = [cs for cs in self._charging_stations if traci.chargingstation.getLaneID(cs) == lane]
        if lane_cs:
            return min(lane_cs, key=lambda cs: abs(traci.chargingstation.getEndPos(cs) - pos))
        else:
            # If no CS on the same lane, find the nearest CS by Euclidean distance
            v_pos = traci.lane.getShape(lane)[int(pos)]
            cs_positions = [traci.chargingstation.getPosition(cs) for cs in self._charging_stations]
            distances = cdist([v_pos], cs_positions)
            return self._charging_stations[np.argmin(distances)]

    def find_nearest_5_cs(self, start_cs):
        sorted_cs = sorted(self._charging_stations, 
                           key=lambda cs: self._time_matrix[start_cs][cs] if cs != start_cs else 0)
        return sorted_cs[:5]

    def calculate_distance(self, lane, pos, cs):
        cs_lane = traci.chargingstation.getLaneID(cs)
        cs_pos = traci.chargingstation.getEndPos(cs)
        if lane == cs_lane:
            return abs(cs_pos - pos)
        else:
            # If on different lanes, use Euclidean distance as an approximation
            v_pos = traci.lane.getShape(lane)[int(pos)]
            cs_pos = traci.chargingstation.getPosition(cs)
            return np.linalg.norm(np.array(v_pos) - np.array(cs_pos))




