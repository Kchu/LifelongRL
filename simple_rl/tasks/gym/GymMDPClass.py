'''
GymMDPClass.py: Contains implementation for MDPs of the Gym Environments.
'''

# Python imports.
import random
import sys
import os
import random

# Other imports.
import gym
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.gym.GymStateClass import GymState

class GymMDP(MDP):
    ''' Class for Gym MDPs '''

    def __init__(self, env_name='CartPole-v0', render=False):
        '''
        Args:
            env_name (str)
        '''
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.render = render
        MDP.__init__(self, range(self.env.action_space.n), self._transition_func, self._reward_func, init_state=GymState(self.env.reset()))
    
    def _reward_func(self, state, action):
        '''
        Args:
            state (AtariState)
            action (str)

        Returns
            (float)
        '''
        obs, reward, is_terminal, info = self.env.step(action)

        if self.render:
            self.env.render()

        self.next_state = GymState(obs, is_terminal=is_terminal)

        return reward

    def _transition_func(self, state, action):
        '''
        Args:
            state (AtariState)
            action (str)

        Returns
            (State)
        '''
        return self.next_state

    def reset(self):
        self.env.reset()

    def __str__(self):
        return "gym-" + str(self.env_name)
