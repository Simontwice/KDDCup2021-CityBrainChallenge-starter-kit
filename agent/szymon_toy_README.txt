The whole idea is to use cycles as the basic building block. A cycle (self.current_phase_plan in the code) is a vector of length self.max_phase (max_phase indicates how many phases we use, currently its only the first 4 out of the whole 8). Every value of a given cycle corresponds to the duration of the respective phase (position 0 -> phase 1, position 1->phase 2 etc.)

It is assumed that a full intersection (4 way intersection) has a cycle of length self.default_cycle_duration. A 3 way intersection will have a cycle that sums up to 3/4 of that time, for obvious reasons - we don't want the 3legged ones to have disproportionally long phase durations.

It is often not the case that lanes corresponding to all 4 phases have vehicles present - in that case we only give positive phase times to lanes with vehicles present. It also means that if, say, only 2 out of 4 phases currently have vehicles, the whole cycle will last 2/4 * self.default_cycle_duration.

After choosing the next phase duration in a given cycle, the corresponding position in the cycle will have its value changed to -10, to indicate that it's already been processed.

NOTE: I used plan and cycle interchangably in the code. Sorry for that.

The cycle for a given agent is self.agent_phase_plan. 
self.change_step[agent] keeps track of when to use another phase from the current cycle
self.next_cycle_plan[agent] keeps track of when to create a new cycle/plan for a given intersection
