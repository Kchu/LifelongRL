''' FourRoomMDPClass.py: Contains the FourRoom class. '''

# Python imports.
import math

# Other imports
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.grid_world.GridWorldMDPClass import GridWorldMDP
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState

class FourRoomMDP(GridWorldMDP):
    ''' Class for a FourRoom '''

    def __init__(self, width=9, height=9, init_loc=(1,1), goal_locs=[(9,9)], gamma=0.99, slip_prob=0.00, name="four_room", is_goal_terminal=True, rand_init=False, step_cost=0.0):
        '''
        Args:
            height (int)
            width (int)
            init_loc (tuple: (int, int))
            goal_locs (list of tuples: [(int, int)...])
        '''
        GridWorldMDP.__init__(self, width, height, init_loc, goal_locs=goal_locs, walls=self._compute_walls(width, height), gamma=gamma, slip_prob=slip_prob, name=name, is_goal_terminal=is_goal_terminal, rand_init=rand_init, step_cost=step_cost)

    def _compute_walls(self, width, height):
        '''
        Args:
            width (int)
            height (int)

        Returns:
            (list): Contains (x,y) pairs that define wall locations.
        '''
        walls = []

        half_width = math.ceil(width / 2.0)
        half_height = math.ceil(height / 2.0)


        # # Wall from left to middle.
        for i in range(1, width + 1):
            if i == half_width:
                half_height -= 1
            if i == (width + 1) / 3 or i == math.ceil(2 * (width + 1) / 3.0):
                continue

            walls.append((i, half_height))

        # Wall from bottom to top.
        for j in range(1, height + 1):
            if j == (height + 1) / 3 or j == math.ceil(2 * (height + 1) / 3.0):
                continue
            walls.append((half_width, j))
        # print(walls)
        # walls = [(1,6), (3,6), (4,6), (5,6), (6,6), (6,7), (6,8), (6,10), (6,11),\
        #          (6,5), (6,4), (6,3), (6,1), (7,5), (8,5), (10,5), (11,5)]
        # walls = [(1,8),(2,8),(3,8),(4,8),(6,8),(7,8),(8,8),(9,8),(11,8),(12,8),(13,8),(14,8),\
        #         (8,1),(8,2),(8,3),(8,4),(8,6),(8,7),(8,8),(8,9),(8,11),(8,12),(8,13),(8,14)]
        # walls = [(1,9),(2,9),(3,9),(4,9),(5,9),(7,9),(8,9),(9,9),(10,9),(11,9),(13,8),(14,8),(15,8),(16,8),(17,8),\
        #         (9,1),(9,2),(9,3),(9,4),(9,5),(9,7),(9,8),(9,9),(9,10),(9,11),(9,13),(9,14),(9,15),(9,16),(9,17)]
        return walls
