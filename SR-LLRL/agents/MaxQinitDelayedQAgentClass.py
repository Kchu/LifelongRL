''' QLearningAgentClass.py: Class for a basic QLearningAgent '''

# Python imports.
import random
import numpy
import time
import copy
from collections import defaultdict

# Other imports.
from simple_rl.agents.AgentClass import Agent
from .QLearningAgentClass import QLearningAgent
from simple_rl.planning.ValueIterationClass import ValueIteration


class MaxQinitDelayedQAgent(Agent):
    '''
    Delayed-Q Learning Agent (Strehl, A.L., Li, L., Wiewiora, E., Langford, J. and Littman, M.L., 2006. PAC model-free reinforcement learning).
    Implemented by Yuu Jinnai (ddyuudd@gmail.com)
    '''

    def __init__(self, actions, init_q=None, default_q=1.0/(1.0-0.99), name="MaxQinit-delayed-Q-learning", gamma=0.99, m=1, epsilon1=0.1, qstar_transfer=False, num_sample_tasks=20, sample_with_q=False):
        '''
        Args:
            actions (list): Contains strings denoting the actions.
            init_q (2d list): Initial Q function. AU(s, a) in Strehl et al 2006.
            name (str): Denotes the name of the agent.
            gamma (float): discount factor
            m (float): Number of samples for updating Q-value
            epsilon1 (float): Learning rate
        '''
        # name_ext = "-" + explore if explore != "uniform" else ""
        Agent.__init__(self, name=name, actions=actions, gamma=gamma)
        self.rmax = 1  # TODO: set/get function

        # Set/initialize parameters and other relevant classwide data
        self.step_number = 0
        self.init_q = defaultdict(lambda : defaultdict(lambda: default_q)) if init_q is None else init_q
        self.default_q = default_q

        # TODO: Here we assume that init_q has Qvalue for every (s, a) pair.
        self.default_q_func = copy.deepcopy(self.init_q)
        self.q_func = copy.deepcopy(self.init_q)

        self.AU = defaultdict(lambda: defaultdict(lambda: 0.0))  # used for attempted updates
        self.l = defaultdict(lambda: defaultdict(lambda: 0))  # counters
        self.b = defaultdict(lambda: defaultdict(lambda: 0))  # beginning timestep of attempted update
        self.LEARN = defaultdict(lambda: defaultdict(lambda: True))  # beginning timestep of attempted update
        # for x in self.init_q:
        #     for y in self.init_q[x]:
        #         self.AU[x][y] = 0.0  # AU(s, a) <- 0
        #         self.l[x][y] = 0  # l(s, a) <- 0
        #         self.b[x][y] = 0  # b(s, a) <- 0
        #         self.LEARN[x][y] = False

        # TODO: Add a code to calculate m and epsilon1 from epsilon and delta.
        # m and epsilon1 should be set according to epsilon and delta in order to be PAC-MDP.
        self.m = m
        self.epsilon1 = epsilon1
        
        self.tstar = 0  # time of most recent action value change
        self.task_number = 1
        self.num_sample_tasks = num_sample_tasks
        self.qstar_transfer = qstar_transfer
        self.sample_with_q = sample_with_q

        self.count_sa = defaultdict(lambda : defaultdict(lambda: 0))
        self.count_s= defaultdict(lambda : 0)
        self.episode_count = defaultdict(lambda : defaultdict(lambda: defaultdict(lambda: 0)))
        self.episode_reward = defaultdict(lambda: 0)

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
        # if self.sample_with_q and self.task_number < self.num_sample_tasks:
        #     return self.q_agent.act(state, reward, learning)
        
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

        if self.LEARN[state][action] == True:
            self.l[state][action] = self.l[state][action] + 1
            nextq, _ = self._compute_max_qval_action_pair(next_state)
            self.AU[state][action] = self.AU[state][action] + reward + self.gamma * nextq 
            # print('AU',self.AU[state][action])
            # print('q',self.q_func[state][action])     
            if self.l[state][action] == self.m:
                if self.q_func[state][action] - self.AU[state][action] / self.m >= 2 * self.epsilon1:
                    self.q_func[state][action] = self.AU[state][action] / self.m + self.epsilon1
                    # print(self.q_func[state][action])
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
            if q_s_a > max_q_val:
                max_q_val = q_s_a
                best_action = action

        return max_q_val, best_action

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

    def reset(self, mdp=None):
        self.step_number = 0
        self.episode_number = 0
        self.AU = defaultdict(lambda: defaultdict(lambda: 0.0))  # used for attempted updates
        self.l = defaultdict(lambda: defaultdict(lambda: 0))  # counters
        self.b = defaultdict(lambda: defaultdict(lambda: 0))  # beginning timestep of attempted update
        self.LEARN = defaultdict(lambda: defaultdict(lambda: True))  # beginning timestep of attempted update
        self.update_init_q_function()
        if self.task_number < self.num_sample_tasks:
            self.q_func = copy.deepcopy(self.init_q)
        else:
            self.q_func = copy.deepcopy(self.default_q_func)
        self.task_number = self.task_number + 1
        Agent.reset(self)

    def end_of_episode(self):
        '''
        Summary:
            Resets the agents prior pointers.
        '''
        Agent.end_of_episode(self)

    def set_q_function(self, q_func):
        '''
        Set initial Q-function.
        For PAC-MDP, initial Q(s, a) should be an upper bound of Q*(s, a).
        '''
        self.init_q_func = copy.deepcopy(q_func)
        self.q_func = copy.deepcopy(self.init_q_func)

    def set_vmax(self, vmax):
        '''
        Initialize Q-values to be Vmax.
        '''
        self.default_q = vmax
        self.q_func = defaultdict(lambda: defaultdict(lambda: vmax))
        self.init_q_func = defaultdict(lambda: defaultdict(lambda: vmax))

    def update_init_q_function(self):
        '''
        If sample_with_q is True, run Q-learning for sample tasks.
        If qstar_transfer is True, run value iteration for sample tasks to get Q*.
        Else, run delayed Q-learning for sample tasks
        '''
        if self.task_number == 1:
            self.default_q_func = defaultdict(lambda: defaultdict(lambda: float("-inf")))
        elif self.task_number < self.num_sample_tasks:
        # if self.task_number < self.num_sample_tasks:
            new_q_func = self.q_func
            for x in new_q_func:
                assert len(self.default_q_func[x]) <= len(new_q_func[x])
                for y in new_q_func[x]:
                    self.default_q_func[x][y] = max(new_q_func[x][y], self.default_q_func[x][y])
                    assert(self.default_q_func[x][y] <= self.default_q)


    def print_dict(self, dic):
        for x in dic:
            for y in dic[x]:
                print("%.2f" % dic[x][y], end='')
            print("")


