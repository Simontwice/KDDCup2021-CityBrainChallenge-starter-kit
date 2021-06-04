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

class TestAgent():
    def __init__(self):
        self.now_phase = {}
        self.base_cycle_duration = 10*4.5
        self.default_cycle_duration = self.base_cycle_duration # duration of 1 CYCLE (also called 1 plan) that will be partitioned according to needs on the intersect
        self.max_phase = 4  # only using the first 4 phases so far
        self.next_cycle_plan = {}  # the time of setting the next cycle
        self.regularisation = 0  # redundant
        self.next_change_step = {}  # the starting time of next phase in the current cycle
        right_turners = [3, 6, 9,
                         12]  # IDs of lanes that turn right on a given intersection (in terms of position in the "observations"
        self.lanes_by_phase = {1: [1, 7], 2: [2, 8], 3: [4, 10],
                               4: [5, 11]}  # IDs of lanes that are involved in phase 1, phase 2,...
        self.agent_phase_plan = {}  # plan for the current cycle, which is a vector of seconds, say: [10,-2,100,40] means that phase 1 gets 10 seconds, phase 3 gets 100, negatives mean no time
        self.agent_list = []
        self.phase_passablelane = {}
        self.intersections = {}
        self.roads = {}
        self.agents = {}
        self.decay_interval = 1200
        self.decay_multiplier = 1.35

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

    ################################

    def default_cycle_time_decay(self, current_time):  # experimental: alter the def cycle duration as time progresses
        time = 30
        if current_time > 1200:
            time += 10
            if current_time > 2400:
                time += 10
        self.default_cycle_duration = 50
        # self.default_cycle_duration = self.base_cycle_duration * (self.decay_multiplier ** int(current_time/self.decay_interval))


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
                now_step = v[0]  # time passed since beginning of sim
                break
        # update default duration of cycles
        self.default_cycle_time_decay(now_step)

        # get actions
        for agent in self.agent_list:
            if agent not in self.next_cycle_plan:  # if we don't have the next_cycle_plan dict, only happens at the beginning
                self.next_cycle_plan[agent] = -1
            if agent not in self.agent_phase_plan:  # if we don't have plans, also only at the beginning, put a dummy plan that will immediately be changed
                self.agent_phase_plan[agent] = np.asarray([-100])

            no_cars = False

            if now_step >= self.next_cycle_plan[agent]:  # if it's time to get a new plan or all the values in the plan are negative
                # GET A NEW PLAN OF PHASE DURATIONS
                vector_of_congestion = observations_for_agent[agent][
                    "lane_vehicle_num"]  # get number of vehicles on every lane , the incoming lanes are on positions 1-12
                phases = []
                for i in range(1, self.max_phase + 1):
                    relevants = [vector_of_congestion[j] for j in self.lanes_by_phase[i]]
                    np_rel = np.asarray(relevants)
                    np_rel[np_rel<=0]=0 # some values will be negative since there are three-legged interestions, and they put negative values in those places
                    suma = np.sum(np_rel)
                    phases.append(suma)
                if max(phases) <= 0:
                    no_cars = True
                if not no_cars:
                    np_phases = np.asarray(phases)
                    sumo = np.sum(np_phases)
                    np_phases = np_phases / sumo  # normalise
                    np_phases *= self.default_cycle_duration*(np.count_nonzero(np_phases)/self.max_phase) # stretch over def_cycle_duration
                    np_phases = smooth_phase_times(np_phases)
                    this_cycle_duration = np.sum(np_phases)+np.count_nonzero(np_phases)*5 #mind the 5 sec all red intermissions!
                    self.agent_phase_plan[agent] = np_phases
                    self.next_change_step[agent] = now_step-1  # its a new plan, time to use it immediately (described below)
                    self.next_cycle_plan[agent] = now_step + this_cycle_duration

            if no_cars:
                self.next_cycle_plan[agent] = now_step-1 # change plan v quickly, cars will probably appear shortly
            # EXECUTE PLAN
            if now_step > self.next_change_step[agent]:  # if it's time to change phases according to the current plan
                if np.max(self.agent_phase_plan[agent]) > 0:
                    argmax_of_list = (self.agent_phase_plan[agent]).argmax() #get the biggest one from the plan
                    new_phase = argmax_of_list + 1 # since phase 1 is on position 0, phase 2 on pos 1 etc.
                    prev_phase = self.now_phase[agent]
                    self.now_phase[agent] = new_phase
                    self.next_change_step[agent] = now_step + np.max(self.agent_phase_plan[agent])
                    self.agent_phase_plan[agent][argmax_of_list] = -10 #set the used phase to -10 to indicate that it's used
                    if self.now_phase[agent] != prev_phase:
                        actions[agent] = self.now_phase[agent]
                else:
                    self.next_cycle_plan[agent] = now_step #if all the phases had neg values, time to get a new plan

        return actions


scenario_dirs = [
    "test"
]

agent_specs = dict.fromkeys(scenario_dirs, None)
for i, k in enumerate(scenario_dirs):
    # initialize an AgentSpec instance with configuration
    agent_specs[k] = TestAgent()


