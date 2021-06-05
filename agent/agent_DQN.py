""" Required submission file

In this file, you should implement your `AgentSpec` instance, and **must** name it as `agent_spec`.
As an example, this file offers a standard implementation.
"""

import pickle
import os
path = os.path.split(os.path.realpath(__file__))[0]
import sys
sys.path.append(path)
import random
from sys import getsizeof

import gym

from pathlib import Path
import pickle
import gym

import tensorflow as tf
import keras
from keras import backend as K

import os
from collections import deque
import numpy as np


# contains all of the intersections
#this matrix sends MODEL type phases to MODEL type vectors
PHASE_TO_VEC = np.array([[1,0,0,0,1,0,0,0],
    [0,1,0,0,1,0,0,0],
    [1,0,0,0,0,1,0,0],
    [0,1,0,0,0,1,0,0],
    [0,0,1,0,0,0,1,0],
    [0,0,0,1,0,0,1,0],
    [0,0,1,0,0,0,0,1],
    [0,0,0,1,0,0,0,1]])

INTERFACE = {0:4,1:6,2:8,3:3,4:2,5:7,6:5,7:1}
INV_INTERFACE = {4:0,6:1,8:2,3:3,2:4,7:5,5:6,1:7}

class TestAgent():
    def __init__(self):

        # DQN parameters
        self.memory = deque(maxlen=2000)
        self.learning_start = 2000
        self.update_model_freq = 20
        self.gamma = 0.97  # discount rate
        self.epsilon = 0.1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.action_space = 8

        # Remember to uncomment the following lines when submitting, and submit your model file as well.
        # path = os.path.split(os.path.realpath(__file__))[0]
        # self.load_model(path, 99)

        self.now_step = 0
        self.current_phase = {} #PHASES AS SEEN BY MODEL, NOT ENVIRONMENT!!!!! SO REFER TO THE PAPER
        self.delay = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.max_phase = 4  # only using the first 4 phases so far
        self.next_cycle_plan = {}  # the time of setting the next cycle
        self.next_change_step = {}  # the starting time of next phase in the current cycle
        right_turners = [3, 6, 9,
                         12]  # IDs of lanes that turn right on a given intersection (in terms of position in the "observations"
        self.lanes_by_phase = {1: [1, 7], 2: [2, 8], 3: [4, 10],
                               4: [5, 11], 5: [1, 2], 6: [4, 5], 7: [7, 8],
                               8: [10, 11]}  # IDs of lanes that are involved in phase 1, phase 2,...
        self.suitable_phases = {"full": [1, 2, 3, 4], "north": [1, 4, 6], "east": [2, 3, 7], "south": [1, 4, 8],
                                "west": [2, 3, 5]}
        self.agent_phase_plan = {}  # plan for the current cycle, which is a vector of seconds, say: [10,-2,100,40] means that phase 1 gets 10 seconds, phase 3 gets 100, negatives mean no time
        self.agent_topology = {}
        self.agent_list = []
        self.old_reward_list = {}
        self.intersections = {}
        self.roads = {}
        self.agents = {}
        self.road_map = {}
        self.model_name = "none"


    ################################
    def load_info(self,args):
        self.memory = deque(maxlen = args.memory_size)
        self.model_name = args.model_name
        self.batch_size = args.batch_size
        self.roadnet_size = args.roadnet_size
        self.epsilon = args.epsilon
        self.learning_rate = args.learning_rate
        self.learning_start = args.memory_size
        self.update_model_freq = args.update_freq
        self.gamma = args.gamma
        self.close_value = args.close_value

    def agent_init(self,args):
        self.load_info(args)
        self.model_A, self.model_B = self._build_model()

    # don't modify this function.
    # agent_list is a list of agent_id
    def load_agent_list(self,agent_list):
        self.agent_list = agent_list
        for id in agent_list:
            self.old_reward_list[id]=0
        self.now_phase = dict.fromkeys(self.agent_list,1)
        self.last_change_step = dict.fromkeys(self.agent_list,0)
    # intersections[key_id] = {
    #     'have_signal': bool,
    #     'end_roads': list of road_id. Roads that end at this intersection. The order is random.
    #     'start_roads': list of road_id. Roads that start at this intersection. The order is random.
    #     'lanes': list, contains the lane_id in. The order is explained in Docs.
    # }
    # roads[road_id] = {
    #     'start_inter':int. Start intersection_id.
    #     'end_inter':int. End intersection_id.
    #     'length': float. Road length.
    #     'speed_limit': float. Road speed limit.
    #     'num_lanes': int. Number of lanes in this road.
    #     'inverse_road':  Road_id of inverse_road.
    #     'lanes': dict. roads[road_id]['lanes'][lane_id] = list of 3 int value. Contains the Steerability of lanes.
    #               lane_id is road_id*100 + 0/1/2... For example, if road 9 have 3 lanes, then their id are 900, 901, 902
    # }
    # agents[agent_id] = list of length 8. contains the inroad0_id, inroad1_id, inroad2_id,inroad3_id, outroad0_id, outroad1_id, outroad2_id, outroad3_id
    def load_roadnet(self, intersections, roads, agents, road_path):
        self.intersections = intersections
        self.roads = roads
        self.agents = agents
        self.usable_phases = {}
        for agent in self.agent_list:
            self.agent_topology[agent] = self.which_topology((agents[agent])[0:4])
            self.current_phase[agent] = 7
        self.load_road_mapping()
        self.load_traffic_signal_dataset(road_path)
        self.inverse_traffic_signal_dataset_2()

    ####################################################################################################################
    def find_last_line(self,path):
        with open(path) as f:
            for i,line in enumerate(reversed(f.readlines())):
                if len(line.split(' '))==1:
                    return -(i)

    def get_road_length(self, id):
        return self.roads[int(id)]['length']

    def which_topology(self, vect):
        vect = np.asarray(vect)
        a = vect < 0
        if a[0]:
            return "north"
        if a[1]:
            return "east"
        if a[2]:
            return "south"
        if a[3]:
            return "west"
        return "full"

    def load_road_mapping(self):
        for road_id, val in self.roads.items():
            d = [val['end_inter'],val['start_inter']]
            for id in d:
                if id not in self.road_map.keys():
                    self.road_map[id] = {
                        'exit_order': [],
                        'approach_order': []
                    }
                if road_id not in self.road_map[id].keys():
                    self.road_map[id][road_id] = {
                        'length': val['length'],
                        'speed_limit': val['speed_limit'],
                        'num_lanes': len(val['lanes'].keys()),
                    }
                    for lane in val['lanes'].keys():
                        lane = int(str(int(lane))[-1])
                        self.road_map[id][road_id][lane] = {
                            'waiting': deque([0, ], maxlen=2),
                            'approaching_close': deque([0, ], maxlen=2),
                            'approaching_far': deque([0, ], maxlen=2),
                            'inv_close': deque([0, ], maxlen=2),
                            'inv_far': deque([0, ], maxlen=2),
                            'wait_time': deque([0, ], maxlen=5),
                        }


    def load_traffic_signal_dataset(self,road_path):
        with open(road_path) as f:
            for line in (f.readlines()[self.find_last_line(road_path):]):
                values = line.split(' ')
                intersection_id = int(values[0])
                roads_id = [int(x) for x in values[1:]]
                self.road_map[intersection_id]['exit_order'] = roads_id

    def inverse_traffic_signal_dataset_2(self):
        for intersection_id, val in self.road_map.items():
            temp = []
            for i in val['exit_order']:
                if i != -1:
                    temp.append(self.roads[i]['inverse_road'])
                else:
                    temp.append(-1)
            self.road_map[intersection_id]['approach_order'] = temp

    def vehicle_info_map(self, info):
        vehicle_info = {}
        for _, val in info.items():

            road_id = int(val['road'][0])
            lane_id = int(val['drivable'][0])
            lane_dir = int(str(lane_id)[-1])
            intersection_id = self.roads[road_id]['end_inter']
            passed_intersection_id = self.roads[road_id]['start_inter']


            # creating keys if key is missing
            if intersection_id not in vehicle_info.keys():
                vehicle_info[intersection_id] = {}
            if road_id not in vehicle_info[intersection_id].keys():
                vehicle_info[intersection_id][road_id] = {}
            if lane_dir not in vehicle_info[intersection_id][road_id].keys():
                vehicle_info[intersection_id][road_id][lane_dir] = {'waiting': 0, 'approaching_close': 0,
                                                                    'approaching_far': 0,'inv_waiting': 0,
                                                                    'inv_close': 0,
                                                                    'inv_far': 0}
            if passed_intersection_id not in vehicle_info.keys():
                vehicle_info[passed_intersection_id] = {}
            if road_id not in vehicle_info[passed_intersection_id].keys():
                vehicle_info[passed_intersection_id][road_id] = {}
            if lane_dir not in vehicle_info[passed_intersection_id][road_id].keys():
                vehicle_info[passed_intersection_id][road_id][lane_dir] = {'waiting': 0, 'approaching_close': 0,
                                                                    'approaching_far': 0,'inv_waiting': 0,
                                                                    'inv_close': 0,
                                                                    'inv_far': 0}

            # calculating waiting cars by speed <= 0
            if val['speed'][0] <= 0:
                #assert(self.get_road_length(val['road'][0]) - val['distance'][0]<2*self.close_value)
                vehicle_info[intersection_id][road_id][lane_dir]['waiting'] += 1

            else:
                if self.get_road_length(val['road'][0]) - val['distance'][0] < self.close_value:
                    vehicle_info[intersection_id][road_id][lane_dir]['approaching_close'] += 1

                else:
                    vehicle_info[intersection_id][road_id][lane_dir]['approaching_far'] += 1

            if val['distance'][0] < self.close_value:
                vehicle_info[passed_intersection_id][road_id][lane_dir]['inv_close']+=1
            else:
                vehicle_info[passed_intersection_id][road_id][lane_dir]['inv_far']+=1



        return vehicle_info

    def implement_to_road_map(self, vehicle_info):
        for intersection_key in self.road_map.keys():
            for road_key in self.road_map[intersection_key].keys():

                if road_key == 'approach_order' or road_key == 'exit_order':
                    continue

                for lane_key in range(self.road_map[intersection_key][road_key]['num_lanes']):

                    try:
                        new_waiting_car = vehicle_info[intersection_key][road_key][lane_key]['waiting']
                        new_approaching_close_car = vehicle_info[intersection_key][road_key][lane_key][
                            'approaching_close']
                        new_approaching_far_car = vehicle_info[intersection_key][road_key][lane_key]['approaching_far']
                        inv_close_car = vehicle_info[intersection_key][road_key][lane_key][
                            'inv_close']
                        inv_far_car = vehicle_info[intersection_key][road_key][lane_key]['inv_far']
                    except KeyError:
                        new_waiting_car = 0
                        new_approaching_far_car = 0
                        new_approaching_close_car = 0
                        inv_far_car = 0
                        inv_close_car = 0

                    old_wait_time = self.road_map[intersection_key][road_key][lane_key]['wait_time'][-1]
                    new_wait_time = (new_waiting_car * 1) + old_wait_time if new_waiting_car != 0 else old_wait_time

                    self.road_map[intersection_key][road_key][lane_key]['waiting'].append(new_waiting_car)
                    self.road_map[intersection_key][road_key][lane_key]['approaching_close'].append(
                        new_approaching_close_car)
                    self.road_map[intersection_key][road_key][lane_key]['approaching_far'].append(
                        new_approaching_far_car)
                    self.road_map[intersection_key][road_key][lane_key]['inv_close'].append(
                        inv_close_car)
                    self.road_map[intersection_key][road_key][lane_key]['inv_far'].append(
                        inv_far_car)
                    self.road_map[intersection_key][road_key][lane_key]['wait_time'].append(new_wait_time)


    def vector_of_time_waiting(self, agent):
        vector_of_congestion = []
        road_order = self.road_map[agent]['approach_order']
        #print(road_order)
        for road in road_order:
            if road == -1:
                for _ in range(3):
                    vector_of_congestion.append(-1)
            else:
                for lane in range(3):
                    if lane not in self.road_map[agent][road].keys():
                        cars = -1
                    else:
                        cars = self.road_map[agent][road][lane]['wait_time'][-1]
                    vector_of_congestion.append(cars)
        assert(len(vector_of_congestion)==12)
        vector_of_congestion = np.asarray(vector_of_congestion)
        vector_of_congestion[vector_of_congestion<0]=0
        return vector_of_congestion

    def vector_of_waiting(self, agent):
        vector_of_congestion = []
        road_order = self.road_map[agent]['approach_order']
        #print(road_order)
        for road in road_order:
            if road == -1:
                for _ in range(3):
                    vector_of_congestion.append(-1)
            else:
                for lane in range(3):
                    if lane not in self.road_map[agent][road].keys():
                        cars = -1
                    else:
                        cars = self.road_map[agent][road][lane]['waiting'][-1]
                    vector_of_congestion.append(cars)
        assert(len(vector_of_congestion)==12)
        vector_of_congestion = np.asarray(vector_of_congestion)
        vector_of_congestion[vector_of_congestion<0]=0
        return vector_of_congestion

    def vector_of_approaching_close(self, agent):
        vector_of_congestion = []
        road_order = self.road_map[agent]['approach_order']
        for road in road_order:
            if road == -1:
                for _ in range(3):
                    vector_of_congestion.append(-1)
            else:
                for lane in range(3):
                    if lane not in self.road_map[agent][road].keys():
                        cars = -1
                    else:
                        cars = self.road_map[agent][road][lane]['approaching_close'][-1]
                    vector_of_congestion.append(cars)
        assert(len(vector_of_congestion)==12)
        vector_of_congestion = np.asarray(vector_of_congestion)
        vector_of_congestion[vector_of_congestion<0]=0
        return vector_of_congestion

    def vector_of_approaching_far(self, agent):
        vector_of_congestion = []
        road_order = self.road_map[agent]['approach_order']
        for road in road_order:
            if road == -1:
                for _ in range(3):
                    vector_of_congestion.append(-1)
            else:
                for lane in range(3):
                    if lane not in self.road_map[agent][road].keys():
                        cars = -1
                    else:
                        cars = self.road_map[agent][road][lane]['approaching_far'][-1]
                    vector_of_congestion.append(cars)
        assert(len(vector_of_congestion)==12)
        vector_of_congestion = np.asarray(vector_of_congestion)
        vector_of_congestion[vector_of_congestion<0]=0
        return vector_of_congestion

    def vector_of_congestion(self, agent):
        vector_of_congestion = []
        road_order = self.road_map[agent]['approach_order']
        for road in road_order:
            if road == -1:
                for _ in range(3):
                    vector_of_congestion.append(-1)
            else:
                for lane in range(3):
                    if lane not in self.road_map[agent][road].keys():
                        cars = -1
                    else:
                        cars = self.road_map[agent][road][lane]['waiting'][-1] + \
                               self.road_map[agent][road][lane]['approaching_close'][-1]
                               #self.road_map[agent][road][lane]['approaching_far'][-1]
                    vector_of_congestion.append(cars)
        assert(len(vector_of_congestion)==12)
        vector_of_congestion = np.asarray(vector_of_congestion)
        vector_of_congestion[vector_of_congestion<0]=0
        return vector_of_congestion

    def vector_of_congestion_baboon(self, agent):

        return self.vector_of_congestion(agent) + self.vector_of_approaching_far(agent)

#///////////////////////////////////////////////////-------------------------------/////////////////////////////////////

    def inv_vector_close(self, agent):
        vector_of_congestion = []
        road_order = self.road_map[agent]['exit_order']
        for road in road_order:
            if road == -1:
                for _ in range(3):
                    vector_of_congestion.append(-1)
            else:
                for lane in range(3):
                    if lane not in self.road_map[agent][road].keys():
                        cars = -1
                    else:
                        cars = self.road_map[agent][road][lane]['inv_close'][-1]
                    vector_of_congestion.append(cars)
        assert(len(vector_of_congestion)==12)
        vector_of_congestion = np.asarray(vector_of_congestion)
        vector_of_congestion[vector_of_congestion<0]=0
        return vector_of_congestion

    def inv_vector_far(self, agent):
        vector_of_congestion = []
        road_order = self.road_map[agent]['exit_order']
        for road in road_order:
            if road == -1:
                for _ in range(3):
                    vector_of_congestion.append(-1)
            else:
                for lane in range(3):
                    if lane not in self.road_map[agent][road].keys():
                        cars = -1
                    else:
                        cars = self.road_map[agent][road][lane]['inv_far'][-1]
                    vector_of_congestion.append(cars)
        assert(len(vector_of_congestion)==12)
        vector_of_congestion = np.asarray(vector_of_congestion)
        vector_of_congestion[vector_of_congestion<0]=0
        return vector_of_congestion

    def inv_vector_of_congestion(self, agent):
        return self.inv_vector_far(agent) + self.inv_vector_far(agent)
######################################################################################################################

    def reinitLayers(self,model):
        session = K.get_session()
        for layer in model.layers:
            for v in layer.__dict__:
                v_arg = getattr(layer, v)
                if hasattr(v_arg, 'initializer'):
                    initializer_method = getattr(v_arg, 'initializer')
                    initializer_method.run(session=session)
                    #print('reinitializing layer {}.{}'.format(layer.name, v))
#####################################################################################################################
    def process_env_ouput(self):
        """
        works out the env type congestion vectors, as well as model type current phase
        :return: concatenated cong vector and phase - so it is env type vec and model phase
        """
        observations_for_agent = {}
        for id in self.agent_list:
            observations_for_agent[id] = {}
            waiting = self.vector_of_waiting(id)
            approaching = self.vector_of_approaching_close(id)
            current_phase = self.current_phase[id]
            #print("ITS BABAOOOOOOON")
            observations_for_agent[id] = np.concatenate((waiting+approaching, [current_phase]))
        return observations_for_agent

    def calculate_rewards(self):
        new_rewards = {}
        for id in self.agent_list:
            new_wait = np.sum(self.vector_of_time_waiting(id)[[0,1,3,4,6,7,9,10]])
            new_rewards[id] = self.old_reward_list[id] - new_wait + 15
            self.old_reward_list[id] = new_wait
        #rewards = {}
        #for agent in self.agent_list:
        #    rewards[agent] = np.sum(self.inv_vector_close(agent) - self.vector_of_approaching_close(agent))
        #    #print("nagorda: ",str(agent),rewards[agent])
        return new_rewards

    def model_env_interface(self,action):
        """

        :param action MODEL TYPE action
        :return: ENV TYPE action
        """
        return INTERFACE[action]

    def env_model_interface(self,action):
        """

        :param action: ENV TYPE ACTION
        :return: MODEL TYPE action (not string of movements, but an index of picked phase (from A-H, or 0-7))
        """
        return INV_INTERFACE[action]

    def permutation_engine_to_model(self,vector_of_congestion):
        if len(vector_of_congestion.shape) in {0,1}:
            return vector_of_congestion[[10,3,1,6,4,9,7,0]]
        else:
            return vector_of_congestion[:,[10,3,1,6,4,9,7,0]]

    def phase_to_input(self,phase):
        if isinstance( phase, int):
            assert (phase in {0, 1, 2, 3, 4, 5, 6, 7})
            return PHASE_TO_VEC[phase]
        if len(phase.shape) in {0}:
            assert(phase in {0,1,2,3,4,5,6,7})
            #print("UWAGA ZMIENIONE#########################",phase)
            return PHASE_TO_VEC[phase]
            #return PHASE_TO_VEC[phase, :]
        elif len(phase.shape) in {1}:
            a = np.asarray([PHASE_TO_VEC[x, :] for x in phase])
            return a
        else:
            raise NotImplementedError

    def _stack(self, thing):
        if len(thing) == 1:
            return np.stack(thing).reshape(1, -1)
        else:
            return np.stack(thing)

    def act_(self, observations_for_agent):
        # Instead of override, We use another act_() function for training,
        # while keep the original act() function for evaluation unchanged.

        actions = {}
        for agent_id in self.agent_list:
            action = self.get_action(observations_for_agent[agent_id])
            self.current_phase[agent_id] = action
            action = self.model_env_interface(action)
            assert action in {1, 2, 3, 4, 5, 6, 7, 8}
            actions[agent_id] = action
        return actions
    """
    def act(self, obs):
        # Get actions
        for agent in self.agent_list:
            self.epsilon = 0
            actions[agent] = self.get_action(observations_for_agent[agent]['lane_vehicle_num']) + 1

        return actions
    """
    def preprocess_for_step(self,ob):
        """
        for information coming straight from the environment
        :param ob: concatenated vector of congestion (in env notation)and phase number IN MODEL NOTATION
        :return: list of unpacked and translated to model: first is vector of movements
        used by the phase, second is congestion
        """
        if len(ob.shape)==1:
            congestion = ob[:-1]
            phase = ob[-1]
            assert(phase in {0,1,2,3,4,5,6,7})
            a=[self.phase_to_input(phase).reshape(1,-1), self.permutation_engine_to_model(congestion).reshape(1,-1)]
            return a
        elif 1:
            raise NotImplementedError

    def preprocess_for_memory(self,ob):
        """
        for information coming straight from the environment, maybe singular (1D array)
        :param ob: concatenated vector of congestion and phase number (phase in model)
        :return: list of unpacked and translated to model: first is translated congestion, second is vector of movements
        used by the phase
        """
        if len(ob.shape)==1:
            congestion = ob[:-1]
            phase = ob[-1]
            #phase = self.env_model_interface(phase)
            assert(phase in {0,1,2,3,4,5,6,7})
            a=[self.phase_to_input(phase),self.permutation_engine_to_model(congestion)]
            return a
        elif 1:
            raise NotImplementedError

# UWAGA MODEL BIERZE NAJPIERW PHASE VECTOR A POTEM CONGESTION
    def get_action(self, ob):
        """
        :param ob ENV TYPE observations! so concatenated (env type congestion ,env type phase)
        :return: predicted MODEL TYPE phase
        """
        # The epsilon-greedy action selector.
        if np.random.rand() <= self.epsilon:
            return self.sample()
        else:
            ob2 = self.preprocess_for_step(ob)
            act_values = self.model_A.predict(ob2)+self.model_B.predict(ob2)
            #print("act_values: ",act_values[0])
            return np.argmax(act_values[0])

    def sample(self):

        # Random samples
        return np.random.randint(0, self.action_space)

    def _build_model(self):

        # Neural Net for Deep-Q learning Model

        model_A = keras.models.load_model(self.model_name)
        model_B = keras.models.load_model(self.model_name)
        #model.summary()
        return model_A,model_B


    def remember(self, ob, action, reward, next_ob):
        a = self.preprocess_for_memory(ob)
        b = self.preprocess_for_memory(next_ob)
        model_action = self.env_model_interface(action)
        assert(b[0] == self.phase_to_input(model_action)).all()
        self.memory.append((a, model_action, reward, b))

    def memory_extract(self,memory):
        obs = [self._stack([x[0][i] for x in memory]) for i in range(2)]
        next_obs = [self._stack([x[3][i] for x in memory]) for i in range(2)]
        [actions,rewards] = [self._stack([x[i] for x in memory]) for i in range(1,3)]

        return obs, actions, rewards, next_obs

    def replay(self):
        # Update the 2Q network from the memory buffer.
        if np.random.rand() <= 0.5:
            a=[self.model_A,self.model_B]
        else:
            a=[self.model_B,self.model_A]

        if self.batch_size > len(self.memory):
            minibatch = self.memory
        else:
            minibatch = random.sample(self.memory, self.batch_size)

        obs, actions, rewards, next_obs, = self.memory_extract(minibatch)
        target = rewards + self.gamma * np.amax(a[1].predict(next_obs), axis=1)
        target_f = a[0].predict(obs)
        #print(target_f)
        print("how heavy is this?             ",getsizeof(self.memory)/1000000," MB")
        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        a[0].fit(obs, target_f,batch_size = 32, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_model(self, dir="model/dqn", step=0):
        name = "dqn_agent_{}.h5".format(step)
        model_name = os.path.join(dir, name)
        print("load from " + model_name)
        self.model_A.load_weights(model_name)
        self.model_B.load_weights(model_name)


    def save_model(self, dir="model/dqn", step=0):
        name = "dqn_agent_{}.h5".format(step)
        model_name = os.path.join(dir, name)
        self.model_A.save_weights(model_name)

scenario_dirs = [
    "test"
]

agent_specs = dict.fromkeys(scenario_dirs, None)
for i, k in enumerate(scenario_dirs):
    # initialize an AgentSpec instance with configuration
    agent_specs[k] = TestAgent()
    # **important**: assign policy builder to your agent spec
    # NOTE: the policy builder must be a callable function which returns an instance of `AgentPolicy`

