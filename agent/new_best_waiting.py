import pickle
import gym

from pathlib import Path
import pickle
import gym

# how to import or load local files
import os
import sys

path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(path)
import gym_cfg
import numpy as np
from collections import deque

with open(path + "/gym_cfg.py", "r") as f:
    pass

class TestAgent():
    def __init__(self):
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
        self.road_map = {}
        ################################

    # don't modify this function.
    # agent_list is a list of agent_id
    def load_agent_list(self, agent_list):
        self.agent_list = agent_list
        self.now_phase = dict.fromkeys(self.agent_list, 1)
        self.next_change_step = dict.fromkeys(self.agent_list, 0)

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
    def load_roadnet(self, intersections, roads, agents):
        self.intersections = intersections
        self.roads = roads
        self.agents = agents
        self.usable_phases = {}
        for agent in self.agent_list:
            self.agent_topology[agent] = self.which_topology((agents[agent])[0:4])
            self.current_phase[agent] = 1
        self.load_road_mapping()
        self.load_traffic_signal_dataset()
        self.inverse_traffic_signal_dataset_2()

    ################################
    def get_road_length(self, id):
        return self.roads[int(id)]['length']

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

    def load_traffic_signal_dataset(self):
        # print(os.listdir())
        with open('data/roadnet_round2.txt') as f:
            for line in (f.readlines()[-859:]):
                values = line.split(' ')
                # values = list(filter(lambda x: x != '-1', values))
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
            val['approach_order'] = temp

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
                if val['distance'][0] / self.get_road_length(val['road'][0]) < 0.4:
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
        return np.concatenate(([0], np.asarray(vector_of_congestion)))

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
        return np.concatenate(([0], np.asarray(vector_of_congestion)))

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
        return np.concatenate(([0], np.asarray(vector_of_congestion)))

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
                               self.road_map[agent][road][lane]['approaching_close'][-1] + \
                               self.road_map[agent][road][lane]['approaching_far'][-1]
                    vector_of_congestion.append(cars)
        return np.concatenate(([0], np.asarray(vector_of_congestion)))

    def inv_vector_of_waiting(self, agent):
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
                        cars = self.road_map[agent][road][lane]['inv_travel_time_waiting'][-1]
                    vector_of_congestion.append(cars)
        return np.concatenate(([0], np.asarray(vector_of_congestion)))

    def inv_vector_of_approaching_close(self, agent):
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
                        cars = self.road_map[agent][road][lane]['inv_travel_time_approaching_close'][-1]
                    vector_of_congestion.append(cars)
        return np.concatenate(([0], np.asarray(vector_of_congestion)))

    def inv_vector_of_approaching_far(self, agent):
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
                        cars = self.road_map[agent][road][lane]['inv_travel_time_approaching_far'][-1]
                    vector_of_congestion.append(cars)
        return np.concatenate(([0], np.asarray(vector_of_congestion)))

    def inv_vector_of_congestion(self, agent):
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
                        cars = self.road_map[agent][road][lane]['inv_travel_time_waiting'][-1] + \
                               self.road_map[agent][road][lane]['inv_travel_time_approaching_close'][-1] + \
                               self.road_map[agent][road][lane]['inv_travel_time_approaching_far'][-1]

                    vector_of_congestion.append(cars)
        return np.concatenate(([0], np.asarray(vector_of_congestion)))

    def default_car_time_decay(self, current_time):  # experimental: alter the def cycle duration as time progresses
        self.base_car_time = 2 + self.delay[int(current_time / 300)]

    def get_next_move(self, agent):
        index = np.argmax(self.agent_phase_plan[agent])
        agent_topology = self.agent_topology[agent]
        phase_list = self.suitable_phases[agent_topology]
        return phase_list[index]

    def create_new_plan(self, agent, vector_of_congestion):
        phases = []
        no_cars = False
        for i in self.suitable_phases[self.agent_topology[agent]]:
            relevants = [vector_of_congestion[j] for j in self.lanes_by_phase[i]]
            np_rel = np.asarray(relevants)
            np_rel[
                np_rel <= 0] = 0  # some values will be negative since there are three-legged interestions, and they put negative values in those places
            suma = np.sum(np_rel)
            phases.append(suma)
        if max(phases) <= 0:
            no_cars = True
        if not no_cars:
            np_phases = np.asarray(phases)
            np_phases = self.base_car_time * np_phases
            np_phases = self.smooth_phase_times(np_phases)
            this_cycle_duration = np.sum(np_phases) + (
                np.count_nonzero(np_phases)) * 5  # mind the 5 sec all red intermissions!
            # this_cycle_duration -= 5 * int()
            self.suitable_phases[self.agent_topology[agent]][np.argmax(np_phases)]
            self.agent_phase_plan[agent] = np_phases
            self.next_change_step[
                agent] = self.now_step - 1  # its a new plan, time to use it immediately (described below)
            self.next_cycle_plan[agent] = self.now_step + this_cycle_duration
        return no_cars

    def change_phase(self, agent):
        index = np.argmax(self.agent_phase_plan[agent])
        agent_topology = self.agent_topology[agent]
        phase_list = self.suitable_phases[agent_topology]
        next_move = phase_list[index]
        self.current_phase[agent] = next_move
        self.next_change_step[agent] = self.now_step + self.agent_phase_plan[agent][index]
        self.agent_phase_plan[agent][index] = -10  # set the used phase to -10 to indicate that it's been used

    def smooth_phase_times(self, plan):  # round the positive times to the closest multiple of 10
        plan[plan > 0] -= plan[plan > 0] % 10 - 10
        plan[plan <= 0] = 0
        return plan

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

    def act(self, obs):
        """ !!! MUST BE OVERRIDED !!!
        """
        # here obs contains all of the observations and infos

        # observations is returned 'observation' of env.step()
        # info is returned 'info' of env.step()
        observations = obs['observations']
        info = obs['info']
        actions = {}
        # converting vehicle info to self.road_map like format to easier use, and storing in vehicle_info{}
        # implementing vehicle_info to general road_map
        self.implement_to_road_map(self.vehicle_info_map(info))

        # preprocess observations
        observations_for_agent = {}
        for key, val in observations.items():
            observations_agent_id = int(key.split('_')[0])
            observations_feature = key[key.find('_') + 1:]
            if (observations_agent_id not in observations_for_agent.keys()):
                observations_for_agent[observations_agent_id] = {}
            observations_for_agent[observations_agent_id][observations_feature] = val

        for agent in self.agent_list:
            for k, v in observations_for_agent[agent].items():
                self.now_step = v[0]  # time passed since beginning of sim
                break

        self.default_car_time_decay(self.now_step)
        # get actions
        for agent in self.agent_list:
            print self.inv_vector_of_congestion(agent)
            break
        for agent in self.agent_list:

            # ASSURANCE THAT THE NEW QUEUE LENGTHS WORK PERFECTLY
            # assert (observations_for_agent[agent]["lane_vehicle_num"][1:13] == self.vector_of_congestion(agent)[1:13]).all()

            if agent not in self.next_cycle_plan:  # if we don't have the next_cycle_plan dict, only happens at the beginning
                self.next_cycle_plan[agent] = -1
            if agent not in self.agent_phase_plan:  # if we don't have plans, also only at the beginning, put a dummy plan that will immediately be changed
                self.agent_phase_plan[agent] = np.asarray([-100])

            if self.now_step >= self.next_cycle_plan[
                agent]:  # if it's time to get a new plan or all the values in the plan are negative
                # GET A NEW PLAN OF PHASE DURATIONS
                vector_of_congestion = self.vector_of_congestion(agent)
                if self.now_step > 1200:
                    vector_of_congestion = self.vector_of_waiting(agent)
                no_cars = self.create_new_plan(agent, vector_of_congestion)
                if no_cars:
                    self.next_cycle_plan[
                        agent] = self.now_step - 1  # change plan v quickly, cars will probably appear shortly
            # EXECUTE PLAN
            if self.now_step >= self.next_change_step[
                agent]:  # if it's time to change phases according to the current plan
                if np.max(self.agent_phase_plan[agent]) > 0:
                    old_phase = self.current_phase[agent]
                    self.change_phase(agent)
                    new_phase = self.current_phase[agent]
                    if old_phase != new_phase:
                        actions[agent] = self.current_phase[agent]
                else:
                    self.next_cycle_plan[
                        agent] = self.now_step  # if all the phases had neg values, time to get a new plan
        return actions


scenario_dirs = [
    "test"
]

agent_specs = dict.fromkeys(scenario_dirs, None)
for i, k in enumerate(scenario_dirs):
    # initialize an AgentSpec instance with configuration
    agent_specs[k] = TestAgent()
