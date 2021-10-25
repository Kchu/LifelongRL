#!/usr/bin/env python

###########################################################################################
# Implementation of SR-LLRL
# Author for codes: Chu Kun(kun_chu@outlook.com), Abel
# Reference: https://github.com/Kchu
###########################################################################################

# Python imports.
from collections import defaultdict
import numpy as np
import sys
import copy
import argparse

## Experiment imports. (please import simple_rl in this folder!)
# Basic imports
from utils import make_mdp_distr
from simple_rl.mdp import MDP, MDPDistribution
from simple_rl.run_experiments import run_agents_lifelong
from simple_rl.planning.ValueIterationClass import ValueIteration
# MaxQinit Agents
from agents.MaxQinitQLearningAgentClass import MaxQinitQLearningAgent
from agents.MaxQinitDelayedQAgentClass import MaxQinitDelayedQAgent
# LRS Agents
from agents.LRSQLearningAgentClass import LRSQLearningAgent
from agents.LRSDelayedQAgentClass import LRSDelayedQAgent
# Baselines Agents
from agents.QLearningAgentClass import QLearningAgent
from agents.DelayedQAgentClass import DelayedQAgent

def get_q_func(vi):
    state_space = vi.get_states()
    q_func = defaultdict(lambda: defaultdict(lambda: 0)) # Assuming R>=0
    for s in state_space:
        for a in vi.actions:
            q_s_a = vi.get_q_value(s, a)
            q_func[s][a] = q_s_a
    return q_func

def parse_args():
    '''
    Summary:
        Parse all arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-mdp_class", type = str, default = "two_room", nargs = '?', help = "Choose the mdp type (one of {octo, hall, grid, taxi, four_room}).")
    parser.add_argument("-goal_terminal", type = bool, default = True, nargs = '?', help = "Whether the goal is terminal.")
    parser.add_argument("-samples", type = int, default = 40, nargs = '?', help = "Number of samples for the experiment.")
    parser.add_argument("-instances", type = int, default = 5, nargs = '?', help = "Number of instances for the experiment.")
    parser.add_argument("-baselines_type", type = str, default = "delayed-q", nargs = '?', help = "Type of agents: (q, rmax, delayed-q).")
    args = parser.parse_args()

    return args.mdp_class, args.goal_terminal, args.samples, args.instances, args.baselines_type

def compute_optimistic_q_function(mdp_distr, sample_rate=5):
    '''
    Instead of transferring an average Q-value, we transfer the highest Q-value in MDPs so that
    it will not under estimate the Q-value.
    '''
    opt_q_func = defaultdict(lambda: defaultdict(lambda: float("-inf")))
    for mdp in mdp_distr.get_mdps():
        # Get a vi instance to compute state space.
        vi = ValueIteration(mdp, delta=0.0001, max_iterations=1000, sample_rate=sample_rate)
        iters, value = vi.run_vi()
        q_func = get_q_func(vi)
        for s in q_func:
            for a in q_func[s]:
                opt_q_func[s][a] = max(opt_q_func[s][a], q_func[s][a])
    return opt_q_func
    
def main(open_plot=True):
    # Environment setting
    episodes = 100
    steps = 100
    gamma = 0.95
    mdp_size = 11
    mdp_class, is_goal_terminal, samples, instance_number, baselines = parse_args()
    
    # Setup multitask setting.
    mdp_distr = make_mdp_distr(mdp_class=mdp_class, mdp_size = mdp_size, is_goal_terminal=is_goal_terminal, gamma=gamma)
    print(mdp_distr.get_num_mdps())
    actions = mdp_distr.get_actions()

    # Get optimal q-function for ideal agent.
    opt_q_func = compute_optimistic_q_function(mdp_distr)

    # Get maximum possible value an agent can get in this environment.
    best_v = -100  
    for x in opt_q_func:
        for y in opt_q_func[x]:
            best_v = max(best_v, opt_q_func[x][y])
    vmax = best_v
    print("Best Vmax =", vmax)
    
    # Init q-funcion
    vmax_func = defaultdict(lambda: defaultdict(lambda: vmax))
    
    # Different baseline learning algorithms
    if baselines == "q":
        # parameter for q-learning
        eps = 0.1
        lrate = 0.1
        
        # basic q-learning agent
        Baselines = QLearningAgent(actions, gamma=gamma, alpha=lrate, epsilon=eps, name="Baselines")
        
        # basic q-learning agent with vmax initialization
        # pure_ql_agent_opt = QLearningAgent(actions, gamma=gamma, alpha=lrate, epsilon=eps, default_q=vmax, name="VTR")        
        
        # MaxQinit agent
        MaxQinit = MaxQinitQLearningAgent(actions, alpha=lrate, epsilon=eps, gamma=gamma, default_q=vmax, name="MaxQInit")
        
        # Ideal agent
        Ideal = QLearningAgent(actions, init_q=opt_q_func, gamma=gamma, alpha=lrate, epsilon=eps, name="Ideal")

        # ALLRL-RS agent
        ALLRL_RS = LRSQLearningAgent(actions, alpha=lrate, epsilon=eps, gamma=gamma, default_q=vmax, name="ALLRL-RS")
        test_1 = LRSQLearningAgent(actions, alpha=lrate, beta = 0.04, epsilon=eps, gamma=gamma, default_q=vmax, name="LRS_0.04")
        test_2 = LRSQLearningAgent(actions, alpha=lrate, beta = 0.05, epsilon=eps, gamma=gamma, default_q=vmax, name="LRS_0.05")
        test_3 = LRSQLearningAgent(actions, alpha=lrate, beta = 0.03, epsilon=eps, gamma=gamma, default_q=vmax, name="LRS_0.03")

        # agents
        agents = [Ideal, MaxQinit, ALLRL_RS, Baselines]
        agents = [Ideal, MaxQinit, test_1, test_2, test_3]

    elif baselines == "delayed-q":
        # parameter for delayed-q
        torelance = 0.001
        min_experience = 5

        # basic delayed-q agent
        Baselines = DelayedQAgent(actions, init_q=vmax_func, gamma=gamma, m=min_experience, epsilon1=torelance, name="Baselines")

        # MaxQinit agent
        MaxQinit = MaxQinitDelayedQAgent(actions, init_q = vmax_func, default_q=vmax, gamma=gamma, m=min_experience, epsilon1=torelance, name="MaxQInit")
        
        # Ideal agent
        Ideal = DelayedQAgent(actions, init_q=opt_q_func, gamma=gamma, m=min_experience, epsilon1=torelance, name="Ideal")
        
        # ALLRL-RS agent
        ALLRL_RS = LRSDelayedQAgent(actions, init_q = vmax_func, gamma=gamma, default_q=vmax, m=min_experience, epsilon1=torelance, name="ALLRL-RS")

        test_1 = LRSDelayedQAgent(actions, init_q = vmax_func, beta = 0.04, gamma=gamma, default_q=vmax, m=min_experience, epsilon1=torelance, name="LRS_0.04")
        test_2 = LRSDelayedQAgent(actions, init_q = vmax_func, beta = 0.05, gamma=gamma, default_q=vmax, m=min_experience, epsilon1=torelance, name="LRS_0.05")
        test_3 = LRSDelayedQAgent(actions, init_q = vmax_func, beta = 0.03, gamma=gamma, default_q=vmax, m=min_experience, epsilon1=torelance, name="LRS_0.03")

        # agents
        agents = [Ideal, MaxQinit, ALLRL_RS, Baselines]
        # agents = [Ideal]
        agents = [Ideal, MaxQinit, test_1, test_2, test_3]

    else:
        msg = "Unknown type of agent:" + baselines + ". Use -agent_type (q, rmax, delayed-q)"
        assert False, msg

    # Experiment body
    run_agents_lifelong(agents, mdp_distr, samples=samples, episodes=episodes, steps=steps, instances=instance_number, reset_at_terminal=is_goal_terminal, track_disc_reward=False, cumulative_plot=False, open_plot=open_plot)

if __name__ == "__main__":
    open_plot = True
    main(open_plot=open_plot)