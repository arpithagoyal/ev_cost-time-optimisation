from __future__ import absolute_import
from __future__ import print_function

import os
import datetime
from shutil import copyfile

from training_simulation import Mysimulation

from generator import TrafficGenerator
from memory import Memory
from brain import Brain
from dqnAgent import Agent
from utils import import_train_configuration, set_sumo, set_train_path


if __name__ == "__main__":

    config = import_train_configuration(config_file='training_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    path = set_train_path(config['models_path_name'])
    
    all_agents = []
    for b_idx in (0,5):
        all_agents.append(Agent(b_idx, config))

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )

    #simulation class object
    mysim = Mysimulation(
        all_agents,
        TrafficGen,
        sumo_cmd,
        config['max_steps'],
        config['num_states'],
        config['num_actions'],
        config['training_epochs'])
    
    episode = 0
    timestamp_start = datetime.datetime.now()

    while episode < config['total_episodes']:
        print('\n----- Episode', str(episode+1), 'of', str(config['total_episodes']))
        simulation_time, training_time = mysim.run(episode)  # run the simulation
        print('Simulation time:', simulation_time, 's Training time:', training_time, 's')
        episode += 1


    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

    #save models here

    copyfile(src='training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))
