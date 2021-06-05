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


with open(path + "/gym_cfg.py", "r") as f:
    pass


def smooth_phase_times(plan):  # round the positive times to the closest multiple of 10
    plan[plan > 0] -= plan[plan > 0] % 10 - 10
    plan[plan <= 0] = 0
    return plan
def which_topology(vect):
    vect = np.asarray(vect)
    a = vect<0
    if a[0]:
        return "north"
    if a[1]:
        return "east"
    if a[2]:
        return "south"
    if a[3]:
        return "west"
    return "full"


class TestAgent():
    def __init__(self):
        self.now_step = 0
        self.current_phase = {}
        self.base_car_time = 2
        self.delay = [0,0,0,0,0,0,0,0,0,0,0,0]
        self.max_phase = 4  # only using the first 4 phases so far
        self.next_cycle_plan = {}  # the time of setting the next cycle
        self.next_change_step = {}  # the starting time of next phase in the current cycle
        right_turners = [3, 6, 9,
                         12]  # IDs of lanes that turn right on a given intersection (in terms of position in the "observations"
        self.lanes_by_phase = {1: [1, 7], 2: [2, 8], 3: [4, 10],
                               4: [5, 11], 5:[1,2],6:[4,5],7:[7,8],8:[10,11]}  # IDs of lanes that are involved in phase 1, phase 2,...
        self.suitable_phases = {"full":[1,2,3,4], "north": [1,4,6], "east": [2,3,7],"south": [1,4,8], "west": [2,3,5]}
        self.agent_phase_plan = {}  # plan for the current cycle, which is a vector of seconds, say: [10,-2,100,40] means that phase 1 gets 10 seconds, phase 3 gets 100, negatives mean no time
        self.agent_topology = {}
        self.agent_list = []
        self.intersections = {}
        self.roads = {}
        self.agents = {}

    ################################
    # don't modify this function.
    # agent_list is a list of agent_id
    def load_agent_list(self, agent_list):
        self.agent_list = agent_list
        self.current_phase = dict.fromkeys(self.agent_list, 1)
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
            self.agent_topology[agent] = which_topology((agents[agent])[0:4])

    ################################

    def default_car_time_decay(self, current_time):  # experimental: alter the def cycle duration as time progresses
        self.base_car_time = 2+self.delay[int(current_time/300)]

    def get_next_move(self,agent):
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
            np_phases = smooth_phase_times(np_phases)
            this_cycle_duration = np.sum(np_phases) + (
                        np.count_nonzero(np_phases)) * 5  # mind the 5 sec all red intermissions!
            #this_cycle_duration -= 5 * int()
            self.suitable_phases[self.agent_topology[agent]][np.argmax(np_phases)]
            self.agent_phase_plan[agent] = np_phases
            self.next_change_step[agent] = self.now_step - 1  # its a new plan, time to use it immediately (described below)
            self.next_cycle_plan[agent] = self.now_step + this_cycle_duration
        return no_cars

    def change_phase(self,agent):
        index = np.argmax(self.agent_phase_plan[agent])
        agent_topology = self.agent_topology[agent]
        phase_list = self.suitable_phases[agent_topology]
        next_move = phase_list[index]
        self.current_phase[agent] = next_move
        self.next_change_step[agent] = self.now_step + self.agent_phase_plan[agent][index]
        self.agent_phase_plan[agent][index] = -10  # set the used phase to -10 to indicate that it's been used

    def act(self, obs):
        """ !!! MUST BE OVERRIDED !!!
        """
        # here obs contains all of the observations and infos

        # observations is returned 'observation' of env.step()
        # info is returned 'info' of env.step()
        observations = obs['observations']
        info = obs['info']
        actions = {}

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
            if agent not in self.next_cycle_plan:  # if we don't have the next_cycle_plan dict, only happens at the beginning
                self.next_cycle_plan[agent] = -1
            if agent not in self.agent_phase_plan:  # if we don't have plans, also only at the beginning, put a dummy plan that will immediately be changed
                self.agent_phase_plan[agent] = np.asarray([-100])

            if self.now_step >= self.next_cycle_plan[agent]:  # if it's time to get a new plan or all the values in the plan are negative
                # GET A NEW PLAN OF PHASE DURATIONS
                vector_of_congestion = observations_for_agent[agent][
                    "lane_vehicle_num"]  # get number of vehicles on every lane , the incoming lanes are on positions 1-12
                no_cars = self.create_new_plan(agent, vector_of_congestion)
                if no_cars:
                    self.next_cycle_plan[agent] = self.now_step-1 # change plan v quickly, cars will probably appear shortly
            # EXECUTE PLAN
            if self.now_step > self.next_change_step[agent]:  # if it's time to change phases according to the current plan
                if np.max(self.agent_phase_plan[agent]) > 0:
                    old_phase = self.current_phase[agent]
                    self.change_phase(agent)
                    new_phase = self.current_phase[agent]
                    if old_phase != new_phase:
                        actions[agent] = self.current_phase[agent]
                else:
                    self.next_cycle_plan[agent] = self.now_step #if all the phases had neg values, time to get a new plan

        return actions


scenario_dirs = [
    "test"
]

agent_specs = dict.fromkeys(scenario_dirs, None)
for i, k in enumerate(scenario_dirs):
    # initialize an AgentSpec instance with configuration
    agent_specs[k] = TestAgent()

