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
        #get all the charging stations
        self._charging_stations = traci.chargingstation.getIDList()
        #update time and cost matrix
        self.initialize_time_and_cost_matrix()
        self.update_time_cost_matrix()
        #update waiting time for each cs
        self.update_waiting_times()

        self._step = 0
        while (self._step<self._max_steps):
            #get the current state vector of 65
            current_state=self.get_current_state()
            actions=[]
            # predict action from each model and append to action array 
            #simulate all actions
            #calculate reward
            #get next state
            # put in memory
            # call replay function to train model 
            #update target model



    def get_current_state(self):
        # Get all vehicles with charge less than 50%
        low_charge_vehicles = [v for v in traci.vehicle.getIDList() 
                               if float(traci.vehicle.getParameter(v, "actualBatteryCapacity")) / 
                                  float(traci.vehicle.getParameter(v, "maximumBatteryCapacity")) < 0.5]

        # Sort by charge level and get up to 5 lowest
        low_charge_vehicles.sort(key=lambda v: float(traci.vehicle.getParameter(v, "actualBatteryCapacity")))
        target_vehicles = low_charge_vehicles[:5]

        state_vector = []

        for i in range(5):
            if i < len(target_vehicles):
                vehicle = target_vehicles[i]
                v_charge = float(traci.vehicle.getParameter(vehicle, "actualBatteryCapacity"))
                v_lane = traci.vehicle.getLaneID(vehicle)
                v_pos = traci.vehicle.getLanePosition(vehicle)

                nearest_cs = self.find_nearest_cs(v_lane, v_pos)
                nearest_5_cs = self.find_nearest_5_cs(nearest_cs)

                v_distance = self.calculate_distance(v_lane, v_pos, nearest_cs)

                state_vector.extend([v_charge, v_lane, v_distance])

                for cs in nearest_5_cs:
                    cs_cost = float(traci.chargingstation.getParameter(cs, "cost"))
                    cs_power = float(traci.chargingstation.getParameter(cs, "power"))
                    state_vector.extend([cs_cost, cs_power])
            else:
                # Add dummy vehicle with 100% charge and all other values as 0
                state_vector.extend([100, 0, 0])  # 100% charge, dummy lane, 0 distance
                state_vector.extend([0, 0] * 5)  # 5 dummy charging stations with 0 cost and 0 power

        return np.array(state_vector)

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




