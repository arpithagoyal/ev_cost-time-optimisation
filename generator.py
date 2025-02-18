import numpy as np
import math
import traci
import xml.etree.ElementTree as ET
import re
import random

class TrafficGenerator:
    def __init__(self, max_steps, n_cars_generated):
        self._n_cars_generated = n_cars_generated  # how many cars per episode
        self._max_steps = max_steps
        self.laneIds = self._getLaneIds()

    def generate_routefile(self, seed):
        """
        Generation of the route of every car for one episode
        """
        pass

    def generate_chargingstationfile(self, seed):

        """ 
        Generation of charging stations data 
        """

        # We are randomizing max_bat_cap, intial_actual_bat_cap, power_of_cs, cost_per_unit of cs
        power_upper_limit = 50000 #W
        power_lower_limit = 1000    #W
        chargeDelay_upper_limit = 10 #s
        chargeDelay_lower_limit = 1 #s
        cost_upper_limit = 15 #rupees/kWh
        cost_lower_limit = 5 #rupees/kWh


        with open("environment/new_charging.add.xml", "w") as stations:
            
            print("""<?xml version="1.0" encoding="UTF-8"?>

<additional>""", file=stations)
            
            for counter, (laneId, laneLength) in enumerate(self.laneIds):
                startPos = max(10, laneLength * 0.3)  # Start 30% into the lane or at least 10m
                endPos = min(laneLength - 10, laneLength * 0.7)  # End before lane length
                cost = np.rint(random.random()*(cost_upper_limit - cost_lower_limit) + cost_lower_limit)
                power = np.rint(random.random()*(power_upper_limit - power_lower_limit) + power_lower_limit)
                chargeDelay = np.rint(random.random()*(chargeDelay_upper_limit - chargeDelay_lower_limit) + chargeDelay_lower_limit)
                if endPos <= startPos:
                    continue  # Skip this lane if positions are invalid

                print("""   <chargingStation id="cs_{}" lane="{}" startPos="{}" endPos="{}" power="{}" chargeDelay="{}">
                <param key="cost" value="{}"/>
                <param key="power" value="{}"/>
                <param key="chargeDelay" value="{}"/>
                </chargingStation>""".format(counter, laneId, startPos, endPos, power, chargeDelay, cost, power, chargeDelay), file=stations)

            print("</additional>", file=stations)
            

    # def _getLaneIds(self):
    #     # Regular expression to match lane IDs
    #     regex_pattern = r'<lane id="([^"]+)"'


    #     # Parse the XML file
    #     tree = ET.parse('./environment/my_grid.net.xml')
    #     root = tree.getroot()

    #     # Convert XML to string
    #     xml_string = ET.tostring(root, encoding='utf8', method='xml')

    #     xml_data = xml_string.decode('utf-8')

    #     # Find all matches
    #     lane_ids = re.findall(regex_pattern, xml_data)

    #     return lane_ids
    def _getLaneIds(self):
        tree = ET.parse('./environment/my_grid.net.xml')
        root = tree.getroot()

        lane_info = []
        for edge in root.findall("edge"):
            for lane in edge.findall("lane"):
                lane_id = lane.get("id")
                lane_length = float(lane.get("length", "0"))  # Get length, default to 0 if missing
                if lane_length > 0:  # Only use valid lanes
                    lane_info.append((lane_id, lane_length))

        return lane_info  # Returns list of (lane_id, lane_length) pairs

    

if __name__ == "__main__":
    trafficgen=TrafficGenerator(10,10)
    trafficgen.generate_chargingstationfile(5)