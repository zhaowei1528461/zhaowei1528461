import numpy as np
import pandas as pd
from planner.plannerBase import PlannerBase
from utils.observation import Observation
from utils.opendrive2discretenet import parse_opendrive
from typing import List, Tuple
from planner.CVaR.parameters import agent_par
# oo = Observation()
# print(8, oo)
class safedreamer(PlannerBase):
    def __init__(self,input_queue,output_queue):
        self.input_queue = input_queue
        self.output_queue = output_queue
    
    def init(self, scenario_dict):
        # print(scenario_dict)
        self.input_queue.put(['init',scenario_dict])
        # if scenario_dict['num'] % 2 == 0:
        #     self.input_queue.put(['init',scenario_dict])
        # else:
        #     self.input_queue.put(['init',0])
        if agent_par['train']:
            assert self.output_queue.get()[0] == 'init success!'
        else:
            try:
                assert self.output_queue.get()[0] == 'init success!'
            except:
                print('not init success!')

    
    def act(self, observation:Observation):
        # print(1)
        self.input_queue.put(['act', observation])
        # print('planner act put success!')
        k = self.output_queue.get()
        if agent_par['train']:
            control = k[1]
            assert k[0] == 'act success!'
            return control
        else:
            try:
                control = k[1]
                assert k[0] == 'act success!'
                return control
            except:
                return [0,0]
            # # print('planner act success!','k: ', k)
            # control = k[1]
            # # control = [-6, -1]
            # assert k[0] == 'act success!'
            # return control
        