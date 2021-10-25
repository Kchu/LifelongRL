###########################################################################################
# Implementation of Lifelong Reward Shaping (LRS)
# Author for codes: Chu Kun(chukun1997@163.com)
# Reference: 
###########################################################################################

# Python imports.
import random
import numpy
import time
import copy
from collections import defaultdict

# Other imports.
from simple_rl.agents.AgentClass import Agent

class LRSDelayedQAgent(Agent):
    '''
    Delayed-Q Learning Agent (Strehl, A.L., Li, L., Wiewiora, E., Langford, J. and Littman, M.L., 2006. PAC model-free reinforcement learning).
    '''

    def __init__(self, actions, init_q=None, beta=1-0.99, default_q=1.0/(1.0-0.99), name="LRS-delayed-Q-learning", gamma=0.99, m=5, epsilon1=0.1):
        '''
        Args:
            actions (list): Contains strings denoting the actions.
            init_q (2d list): Initial Q function. AU(s, a) in Strehl et al 2006.
            name (str): Denotes the name of the agent.
            gamma (float): discount factor
            m (float): Number of samples for updating Q-value
            epsilon1 (float): Learning rate
        '''
        # Set initial q func.
        self.rmax = 1  # TODO: set/get function
        self.beta = beta
        self.init_q = defaultdict(lambda : defaultdict(lambda: default_q)) if init_q is None else init_q
        self.default_q = default_q
        self.default_q_func = copy.deepcopy(self.init_q)
        self.q_func = copy.deepcopy(self.init_q)

        Agent.__init__(self, name=name, actions=actions, gamma=gamma)

        # Set/initialize parameters and other relevant classwide data
        self.step_number = 0
        self.AU = defaultdict(lambda: defaultdict(lambda: 0.0))  # used for attempted updates
        self.l = defaultdict(lambda: defaultdict(lambda: 0))  # counters
        self.b = defaultdict(lambda: defaultdict(lambda: 0))  # beginning timestep of attempted update
        self.LEARN = defaultdict(lambda: defaultdict(lambda: True))  # beginning timestep of attempted update

        # TODO: Add a code to calculate m and epsilon1 from epsilon and delta.
        # m and epsilon1 should be set according to epsilon and delta in order to be PAC-MDP.
        self.m = m
        self.epsilon1 = epsilon1
        self.tstar = 0  # time of most recent action value change

        # LRS setting
        self.count_sa = defaultdict(lambda : defaultdict(lambda: 0))
        self.reward_sa = defaultdict(lambda : defaultdict(lambda: 0))
        self.count_s= defaultdict(lambda : 0)
        self.episode_count = defaultdict(lambda : defaultdict(lambda: defaultdict(lambda: 0)))
        self.episode_reward = defaultdict(lambda: 0)
        self.task_number = 1
        
    # --------------------------------
    # ---- CENTRAL ACTION METHODS ----
    # --------------------------------

    def act(self, state, reward, learning=True):
        '''
        Args:
            state (State)
            reward (float)

        Summary:
            The central method called during each time step.
            Retrieves the action according to the current policy
            and performs updates given (s=self.prev_state,
            a=self.prev_action, r=reward, s'=state)
        '''
        if learning:
            self.update(self.prev_state, self.prev_action, reward, state)

        # For Delayed Q-learning it always take the action with highest Q value (no epsilon exploration required).
        action = self.greedy_q_policy(state)

        self.prev_state = state
        self.prev_action = action
        self.step_number += 1

        return action

    def greedy_q_policy(self, state):
        '''
        Args:
            state (State)

        Returns:
            (str): action.
        '''
        action = self.get_max_q_action(state)
        return action

    # ---------------------------------
    # ---- Q VALUES AND PARAMETERS ----
    # ---------------------------------

    def update(self, state, action, reward, next_state):
        '''
        Args:
            state (State)
            action (str)
            reward (float)
            next_state (State)

        Summary:
            Updates the internal Q Function according to the Bellman Equation. (Classic Q Learning update)
        '''
        if state is None:
            self.prev_state = next_state
            return

        # if state.is_terminal():
        #     # If the state is terminal we set the Q values to 0
        #     for a in self.actions:
        #         self.q_func[state][a] = 0
        #     return

        # if self.b[state][action] <= self.tstar:
        #     self.LEARN[state][action] = True

        reward = reward + self.reward_sa[state][action]

        if self.LEARN[state][action] == True:
            self.l[state][action] = self.l[state][action] + 1
            nextq, _ = self._compute_max_qval_action_pair(next_state)
            self.AU[state][action] = self.AU[state][action] + reward + self.gamma * nextq 
            if self.l[state][action] == self.m:
                if self.q_func[state][action] - self.AU[state][action] / self.m >= 2 * self.epsilon1:
                    self.q_func[state][action] = self.AU[state][action] / self.m + self.epsilon1
                    self.tstar = self.step_number
                elif self.b[state][action] >= self.tstar:
                    self.LEARN[state][action] = False
                self.b[state][action] = self.step_number
                self.AU[state][action] = 0
                self.l[state][action] = 0
        elif self.b[state][action] < self.tstar:
            self.LEARN[state][action] = True

    def _compute_max_qval_action_pair(self, state):
        '''
        Args:
            state (State)

        Returns:
            (tuple) --> (float, str): where the float is the Qval, str is the action.
        '''
        # Grab random initial action in case all equal
        best_action = random.choice(self.actions)
        max_q_val = float("-inf")
        shuffled_action_list = self.actions[:]
        random.shuffle(shuffled_action_list)

        # Find best action (action w/ current max predicted Q value)
        for action in shuffled_action_list:
            q_s_a = self.get_q_value(state, action)
            # print(q_s_a)
            if q_s_a > max_q_val:
                max_q_val = q_s_a
                best_action = action

        return max_q_val, best_action

    def _compute_count_reward(self):
        for x in self.count_sa:
            for y in self.count_sa[x]:
                self.reward_sa[x][y] = self.beta * ((self.count_sa[x][y] / self.count_s[x])) * self.default_q

    def get_max_q_action(self, state):
        '''
        Args:
            state (State)

        Returns:
            (str): denoting the action with the max q value in the given @state.
        '''
        return self._compute_max_qval_action_pair(state)[1]

    def get_max_q_value(self, state):
        '''
        Args:
            state (State)

        Returns:
            (float): denoting the max q value in the given @state.
        '''
        return self._compute_max_qval_action_pair(state)[0]

    def get_q_value(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns:
            (float): denoting the q value of the (@state, @action) pair.
        '''
        return self.q_func[state][action]

    def get_action_distr(self, state, beta=0.2):
        '''
        Args:
            state (State)
            beta (float): Softmax temperature parameter.

        Returns:
            (list of floats): The i-th float corresponds to the probability
            mass associated with the i-th action (indexing into self.actions)
        '''
        all_q_vals = []
        for i in range(len(self.actions)):
            action = self.actions[i]
            all_q_vals.append(self.get_q_value(state, action))

        # Softmax distribution.
        total = sum([numpy.exp(beta * qv) for qv in all_q_vals])
        softmax = [numpy.exp(beta * qv) / total for qv in all_q_vals]

        return softmax

    def reset(self):
        self.step_number = 0
        self.episode_number = 0
        self.task_number +=1
        self.AU = defaultdict(lambda: defaultdict(lambda: 0.0))  # used for attempted updates
        self.l = defaultdict(lambda: defaultdict(lambda: 0))  # counters
        self.b = defaultdict(lambda: defaultdict(lambda: 0))  # beginning timestep of attempted update
        self.LEARN = defaultdict(lambda: defaultdict(lambda: True))  # beginning timestep of attempted update
        self._compute_count_reward()
        self.q_func = copy.deepcopy(self.default_q_func)
        Agent.reset(self)

    def end_of_episode(self):
        '''
        Summary:
            Resets the agents prior pointers.
        '''
        Agent.end_of_episode(self)
