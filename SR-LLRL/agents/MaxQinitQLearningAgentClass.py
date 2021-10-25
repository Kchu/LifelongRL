''' QLearningAgentClass.py: Class for a basic QLearningAgent '''

# Python imports.
import random
import numpy
import time
import copy
from collections import defaultdict

# Other imports.
from simple_rl.agents.AgentClass import Agent
from simple_rl.planning.ValueIterationClass import ValueIteration


class MaxQinitQLearningAgent(Agent):
    ''' Implementation for a Q Learning Agent '''

    def __init__(self, actions, name="MaxQinit-Q-learning", init_q=None, alpha=0.05, gamma=0.99, epsilon=0.1, explore="uniform", anneal=False, default_q=1.0/(1.0-0.99), num_sample_tasks=20):
        '''
        Args:
            actions (list): Contains strings denoting the actions.
            name (str): Denotes the name of the agent.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Exploration term.
            explore (str): One of {softmax, uniform}. Denotes explore policy.
        '''
        name_ext = "-" + explore if explore != "uniform" else ""
        Agent.__init__(self, name=name + name_ext, actions=actions, gamma=gamma)

        # Set/initialize parameters and other relevant classwide data
        self.alpha, self.alpha_init = alpha, alpha
        self.epsilon, self.epsilon_init = epsilon, epsilon
        self.step_number = 0
        self.anneal = anneal
        init_q = defaultdict(lambda : defaultdict(lambda: default_q)) if init_q is None else init_q
        self.init_q = init_q
        self.default_q = default_q
        
        # Q Function:
        # Key: state
        # Val: dict
        #   Key: action
        #   Val: q-value
        self.q_func = defaultdict(lambda : defaultdict(lambda: self.default_q))
        self.default_q_func = copy.deepcopy(self.q_func)

        # Choose explore type.
        self.explore = explore

        self.task_number = 1
        self.num_sample_tasks = num_sample_tasks

        self.count_sa = defaultdict(lambda : defaultdict(lambda: 0))
        self.count_s= defaultdict(lambda : 0)
        self.episode_count = defaultdict(lambda : defaultdict(lambda: defaultdict(lambda: 0)))
        self.episode_reward = defaultdict(lambda: 0)

    # --------------------------------
    # ---- CENTRAL ACTION METHODS ----
    # --------------------------------

    def act(self, state, reward, explore=True, learning=True):
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
        
        if explore: 
            if self.explore == "softmax":
                # Softmax exploration
                action = self.soft_max_policy(state)
            else:
                # Uniform exploration
                action = self.epsilon_greedy_q_policy(state)
        else:
            action = self.get_max_q_action(state)

        self.prev_state = state
        self.prev_action = action
        self.step_number += 1

        # Anneal params.
        if learning and self.anneal:
            self._anneal()

        return action

    def epsilon_greedy_q_policy(self, state):
        '''
        Args:
            state (State)
        Returns:
            (str): action.
        '''
        # Policy: Epsilon of the time explore, otherwise, greedyQ.
        if numpy.random.random() > self.epsilon:
            # Exploit.
            action = self.get_max_q_action(state)
        else:
            # Explore
            action = numpy.random.choice(self.actions)

        return action

    def soft_max_policy(self, state):
        '''
        Args:
            state (State): Contains relevant state information.
        Returns:
            (str): action.
        '''
        return numpy.random.choice(self.actions, 1, p=self.get_action_distr(state))[0]

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
        # If this is the first state, just return.
        if state is None:
            self.prev_state = next_state
            return

        if state.is_terminal():
            # If the state is terminal we set the Q values to 0
            for a in self.actions:
                self.q_func[state][a] = 0.0
            # print("State is terminal!")
            # print(self.q_func[state])
            return
        
        if next_state.is_terminal():
            # If the state is terminal we set the Q values to 0
            for a in self.actions:
                self.q_func[state][a] = 0.0
            # print("next_state is terminal!")
            # print(self.q_func[state])
            return

        # Update the Q Function.
        max_q_curr_state = self.get_max_q_value(next_state)
        prev_q_val = self.get_q_value(state, action)
        self.q_func[state][action] = (1 - self.alpha) * prev_q_val + self.alpha * (reward + self.gamma*max_q_curr_state)

    def _anneal(self):
        # Taken from "Note on learning rate schedules for stochastic optimization, by Darken and Moody (Yale)":
        self.alpha = self.alpha_init / (1.0 +  (self.step_number / 200.0)*(self.episode_number + 1) / 2000.0 )
        self.epsilon = self.epsilon_init / (1.0 + (self.step_number / 200.0)*(self.episode_number + 1) / 2000.0 )

    def _compute_max_qval_action_pair(self, state):
        '''
        Args:
            state (State)
        Returns:
            (tuple) --> (float, str): where the float is the Qval, str is the action.
        '''
        assert(not state.is_terminal())
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
        for i in xrange(len(self.actions)):
            action = self.actions[i]
            all_q_vals.append(self.get_q_value(state, action))

        # Softmax distribution.
        total = sum([numpy.exp(beta * qv) for qv in all_q_vals])
        softmax = [numpy.exp(beta * qv) / total for qv in all_q_vals]

        return softmax

    def reset(self, mdp=None):
        self.step_number = 0
        self.episode_number = 0
        # if mdp is not None:
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
        if self.anneal:
            self._anneal()
        Agent.end_of_episode(self)

    def set_init_q_function(self, q_func):
        '''
        Function for transferring q function
        '''
        self.default_q_func = copy.deepcopy(q_func)
        self.q_func = copy.deepcopy(self.default_q_func)

    def update_init_q_function(self):
        if self.task_number == 1:
            self.default_q_func = defaultdict(lambda: defaultdict(lambda: float("-inf")))
        elif self.task_number < self.num_sample_tasks:
            # print(self.num_sample_tasks)
            new_q_func = self.q_func
            for x in new_q_func:
                for y in new_q_func[x]:
                    self.default_q_func[x][y] = max(new_q_func[x][y], self.default_q_func[x][y])

    def print_dict(self, dic):
        for x in dic:
            for y in dic[x]:
                print("%.2f" % dic[x][y], end='')
            print("")