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

import gym

from pathlib import Path
import pickle
import gym

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop, SGD
import os
from collections import deque
import numpy as np
from keras.layers.merge import concatenate
from keras.layers import Input, Dense, Conv2D, Flatten
from keras.models import Model

# contains all of the intersections

PHASE_TO_VEC = np.array([[0,0,0,1,0,0,0,1],
    [0,0,1,0,0,0,1,0],
    [0,1,0,0,0,1,0,0],
    [1,0,0,0,1,0,0,0],
    [0,0,1,0,0,0,0,1],
    [0,1,0,0,1,0,0,0],
    [0,0,0,1,0,0,1,0],
    [1,0,0,0,0,1,0,0]])

INTERFACE = {1:4,2:6,3:8,4:3,5:2,6:7,7:5,8:1}
INV_INTERFACE = {4:1,6:2,8:3,3:4,2:5,7:6,5:7,1:8}

class TestAgent():
    def __init__(self):

        # DQN parameters
        self.memory = deque(maxlen=2000)
        self.learning_start = 2000
        self.update_model_freq = 5
        self.update_target_model_freq = 20

        self.gamma = 0.95  # discount rate
        self.epsilon = 0.1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.batch_size = 32
        self.ob_length = 25

        self.action_space = 8

        self.model = self._build_model()

        # Remember to uncomment the following lines when submitting, and submit your model file as well.
        # path = os.path.split(os.path.realpath(__file__))[0]
        # self.load_model(path, 99)
        self.target_model = self._build_model()
        self.update_target_network()

        self.now_step = 0
        self.current_phase = {}
        self.base_car_time = 2
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
        self.intersections = {}
        self.roads = {}
        self.agents = {}
        self.close_value = 50
        self.road_map = {}

    ################################
    # don't modify this function.
    # agent_list is a list of agent_id
    def load_agent_list(self,agent_list):
        self.agent_list = agent_list
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
    def load_roadnet(self, intersections, roads, agents,road_path):
        self.intersections = intersections
        self.roads = roads
        self.agents = agents
        self.usable_phases = {}
        for agent in self.agent_list:
            self.agent_topology[agent] = self.which_topology((agents[agent])[0:4])
            self.current_phase[agent] = 1
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
            intersection_id = val['end_inter']

            if intersection_id not in self.road_map.keys():
                self.road_map[intersection_id] = {
                    'exit_order': [],
                    'approach_order': []
                }
            if road_id not in self.road_map[intersection_id].keys():
                self.road_map[intersection_id][road_id] = {
                    'length': val['length'],
                    'speed_limit': val['speed_limit'],
                    'num_lanes': len(val['lanes'].keys()),
                }
                for lane in val['lanes'].keys():
                    lane = int(str(int(lane))[-1])
                    self.road_map[intersection_id][road_id][lane] = {
                        'waiting': deque([0, ], maxlen=12),
                        'approaching_close': deque([0, ], maxlen=12),
                        'approaching_far': deque([0, ], maxlen=12),
                        'inv_travel_time_waiting': deque([0, ], maxlen=12),
                        'inv_travel_time_approaching_close': deque([0, ], maxlen=12),
                        'inv_travel_time_approaching_far': deque([0, ], maxlen=12),
                        'wait_time': deque([0, ], maxlen=12),
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

            # creating keys if key is missing
            if intersection_id not in vehicle_info.keys():
                vehicle_info[intersection_id] = {}
            if road_id not in vehicle_info[intersection_id].keys():
                vehicle_info[intersection_id][road_id] = {}
            if lane_dir not in vehicle_info[intersection_id][road_id].keys():
                vehicle_info[intersection_id][road_id][lane_dir] = {'waiting': 0, 'approaching_close': 0,
                                                                    'approaching_far': 0,'inv_travel_time_waiting': 0, 'inv_travel_time_approaching_close': 0,
                                                                    'inv_travel_time_approaching_far': 0}

            # calculating waiting cars by speed <= 0
            if val['speed'][0] <= 0:
                vehicle_info[intersection_id][road_id][lane_dir]['waiting'] += 1
                vehicle_info[intersection_id][road_id][lane_dir]['inv_travel_time_waiting'] += 1 / val['t_ff'][0]

            else:
                if self.get_road_length(val['road'][0]) - val['distance'][0] < self.close_value:
                    vehicle_info[intersection_id][road_id][lane_dir]['approaching_close'] += 1
                    vehicle_info[intersection_id][road_id][lane_dir]['inv_travel_time_approaching_close'] += 1 / val['t_ff'][0]

                else:
                    vehicle_info[intersection_id][road_id][lane_dir]['approaching_far'] += 1
                    vehicle_info[intersection_id][road_id][lane_dir]['inv_travel_time_approaching_far'] += 1 / val['t_ff'][0]

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
                        inv_travel_time_new_waiting_car = vehicle_info[intersection_key][road_key][lane_key]['inv_travel_time_waiting']
                        inv_travel_time_new_approaching_close_car = vehicle_info[intersection_key][road_key][lane_key][
                            'inv_travel_time_approaching_close']
                        inv_travel_time_new_approaching_far_car = vehicle_info[intersection_key][road_key][lane_key]['inv_travel_time_approaching_far']
                    except KeyError:
                        new_waiting_car = 0
                        new_approaching_far_car = 0
                        new_approaching_close_car = 0
                        inv_travel_time_new_waiting_car = 0
                        inv_travel_time_new_approaching_far_car = 0
                        inv_travel_time_new_approaching_close_car = 0

                    old_wait_time = self.road_map[intersection_key][road_key][lane_key]['wait_time'][-1]

                    new_wait_time = (new_waiting_car * 10) + old_wait_time if new_waiting_car != 0 else 0

                    self.road_map[intersection_key][road_key][lane_key]['waiting'].append(new_waiting_car)
                    self.road_map[intersection_key][road_key][lane_key]['approaching_close'].append(
                        new_approaching_close_car)
                    self.road_map[intersection_key][road_key][lane_key]['approaching_far'].append(
                        new_approaching_far_car)
                    self.road_map[intersection_key][road_key][lane_key]['inv_travel_time_waiting'].append(inv_travel_time_new_waiting_car)
                    self.road_map[intersection_key][road_key][lane_key]['inv_travel_time_approaching_close'].append(
                        inv_travel_time_new_approaching_close_car)
                    self.road_map[intersection_key][road_key][lane_key]['inv_travel_time_approaching_far'].append(
                        inv_travel_time_new_approaching_far_car)
                    self.road_map[intersection_key][road_key][lane_key]['wait_time'].append(new_wait_time)

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
        return np.asarray(vector_of_congestion)

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
        return np.asarray(vector_of_congestion)

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
        return np.asarray(vector_of_congestion)

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
        return np.asarray(vector_of_congestion)

    ####################################################################################################################
    def model_env_interface(self,action):
        return INTERFACE[action]

    def env_model_interface(self,action):
        return INV_INTERFACE[action]

    def act_(self, observations_for_agent):
        # Instead of override, We use another act_() function for training,
        # while keep the original act() function for evaluation unchanged.

        actions = {}
        for agent_id in self.agent_list:
            action = self.get_action(observations_for_agent[agent_id])
            action = self.model_env_interface(action)
            print(action)
            assert action in {1,2,3,4,5,6,7,8}
            actions[agent_id] = action
        return actions

    def act(self, obs):
        observations = obs['observations']
        info = obs['info']
        actions = {}

        # Get state
        observations_for_agent = {}
        for key,val in observations.items():
            observations_agent_id = int(key.split('_')[0])
            observations_feature = key[key.find('_')+1:]
            if(observations_agent_id not in observations_for_agent.keys()):
                observations_for_agent[observations_agent_id] = {}
            observations_for_agent[observations_agent_id][observations_feature] = val[1:]

        # Get actions
        for agent in self.agent_list:
            self.epsilon = 0
            actions[agent] = self.model_env_interface(self.get_action(observations_for_agent[agent],test=True))

        return actions

    def get_action(self, ob, test=False):

        # The epsilon-greedy action selector.
        if not test:
            if np.random.rand() <= self.epsilon:
                return self.sample()
        k = self.simple_reshape_ob(ob)
        act_values = self.model.predict(k)
        print("ACT_VALUES: ",act_values)
        print(np.argmax(act_values[0]) + 1)
        assert np.argmax(act_values[0])+1>0
        return np.argmax(act_values[0]) + 1

    def sample(self):

        # Random samples

        return np.random.randint(0, self.action_space)

    def _build_model(self):

        # Neural Net for Deep-Q learning Model

        model = keras.models.load_model('keras_model.h5')
        #model.summary()
        return model

    def permutation_engine_to_model(self,vector_of_congestion):
        if len(vector_of_congestion.shape) in {0,1}:
            return vector_of_congestion[[10,3,1,8,6,9,7,0]]
        else:
            return vector_of_congestion[:,[10,3,1,8,6,9,7,0]]
    def phase_to_input(self,phase):
        if len(phase.shape)in {0}:
            return PHASE_TO_VEC[phase,:]
        elif len(phase.shape) in {1}:
            a=np.asarray([PHASE_TO_VEC[x,:] for x in phase])
            return a
        else:
            raise NotImplementedError

    def _stack(self,thing):
        if len(thing)==1:
            return np.stack(thing).reshape(1,-1)
        else:
            return np.stack(thing)

    def simple_reshape_ob(self, ob):
        congestion = ob[:-1]
        phase = ob[-1]
        a=[self.permutation_engine_to_model(congestion).reshape(1,-1), self.phase_to_input(phase).reshape(1,-1)]
        return a

    def _reshape_ob(self, ob):
        #congestion = ob[:,:-1]
        #phase = ob[:,-1]
        #a=[self.permutation_engine_to_model(congestion), self.phase_to_input(phase)]
        #assert a[0].shape[-1] == 8 and a[1].shape[-1] == 8
        #return ob
        return ob

    def update_target_network(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)

    def remember(self, ob, action, reward, next_ob):
        congestion = ob[:-1]
        phase = ob[-1]
        a = [self.permutation_engine_to_model(congestion), self.phase_to_input(phase)]
        congestion = next_ob[:-1]
        phase = next_ob[-1]
        b = [self.permutation_engine_to_model(congestion), self.phase_to_input(phase)]
        self.memory.append((a, action, reward, b))

    def memory_extract(self,memory):
        obs = [self._stack([x[0][i] for x in memory]) for i in range(2)]
        next_obs = [self._stack([x[3][i] for x in memory]) for i in range(2)]
        [actions,rewards] = [self._stack([x[i] for x in memory]) for i in range(1,3)]

        return obs, actions, rewards, next_obs

    def replay(self):
        # Update the Q network from the memory buffer.

        if self.batch_size > len(self.memory):
            minibatch = self.memory
        else:
            minibatch = random.sample(self.memory, self.batch_size)

        obs, actions, rewards, next_obs, = self.memory_extract(minibatch)
        print("ACTIONSEEEEEEEEEEEEEEEEEE",actions)

        target = rewards + self.gamma * np.amax(self.model.predict(self._reshape_ob(obs)))
        target_f = self.model.predict(self._reshape_ob(obs))
        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        self.model.fit(self._reshape_ob(obs), target_f, epochs=1, verbose=True)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_model(self, dir="model/dqn", step=0):
        name = "dqn_agent_{}.h5".format(step)
        model_name = os.path.join(dir, name)
        print("load from " + model_name)
        self.model.load_weights(model_name)

    def save_model(self, dir="model/dqn", step=0):
        name = "dqn_agent_NEW_{}.h5".format(step)
        model_name = os.path.join(dir, name)
        self.model.save_weights(model_name)

scenario_dirs = [
    "test"
]

agent_specs = dict.fromkeys(scenario_dirs, None)
for i, k in enumerate(scenario_dirs):
    # initialize an AgentSpec instance with configuration
    agent_specs[k] = TestAgent()
    # **important**: assign policy builder to your agent spec
    # NOTE: the policy builder must be a callable function which returns an instance of `AgentPolicy`
