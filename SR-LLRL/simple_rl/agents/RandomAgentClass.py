''' RandomAgentClass.py: Class for a randomly acting RL Agent '''

# Python imports.
import random
from collections import defaultdict

# Other imports
from simple_rl.agents.AgentClass import Agent

class RandomAgent(Agent):
    ''' Class for a random decision maker. '''

    def __init__(self, actions, name=""):
        name = "Random" if name is "" else name
        Agent.__init__(self, name=name, actions=actions)
        self.count_sa = defaultdict(lambda : defaultdict(lambda: 0))
        self.count_s= defaultdict(lambda : 0)
        self.episode_count = defaultdict(lambda : defaultdict(lambda: defaultdict(lambda: 0)))
        self.episode_reward = defaultdict(lambda: 0)

    def act(self, state, reward):
        return random.choice(self.actions)
