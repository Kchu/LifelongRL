# Python imports.i
import itertools
import random

# Other imports
from simple_rl.tasks import ChainMDP, GridWorldMDP, TaxiOOMDP, RandomMDP, FourRoomMDP, ComboLockMDP
from simple_rl.tasks.grid_world.GridWorldMDPClass import make_grid_world_from_file
from simple_rl.mdp import MDPDistribution
from simple_rl.tasks.navigation import NavigationMDP

def coord_from_binary_list(l):
    coord = []
    for i in range(len(l)):
        for j in range(len(l[i])):
            if l[i][j] == 1:
                coord.append((j + 1, len(l) - i))
            else:
                if l[i][j] != 0:
                    raise ValueError('List elements should be either 0 or 1.')
    return coord

def make_mdp_distr(mdp_class, is_goal_terminal, mdp_size=11, horizon=0, gamma=0.99):
    '''
    Args:
        mdp_class (str): 
        horizon (int)
        step_cost (float)
        gamma (float)

    Returns:
        (MDPDistribution)
    '''
    mdp_dist_dict = {}

    height, width, = mdp_size, mdp_size

    navigation_goal_locs = [(4,11), (6,11),(8,11),(11,11)]
    # Corridor(走廊).
    corr_width = 20
    corr_goal_magnitude = 1 #random.randint(1, 5)
    corr_goal_cols = [i for i in range(1, corr_goal_magnitude + 1)] + [j for j in range(corr_width-corr_goal_magnitude + 1, corr_width + 1)]
    # corr_goal_cols：[1, 20]
    corr_goal_locs  = list(itertools.product(corr_goal_cols, [1]))
    # corr_goal_locs: [(1, 1), (20, 1)]

    # Grid World
    tl_grid_world_rows, tl_grid_world_cols = [i for i in range(width - 4, width)], [j for j in range(height - 4, height)]
    # tl_grid_world_rows, tl_grid_world_cols: [7, 8, 9, 10]
    tl_grid_goal_locs = list(itertools.product(tl_grid_world_rows, tl_grid_world_cols))
    # tl_grid_goal_locs: len(16)
    tr_grid_world_rows, tr_grid_world_cols = [i for i in range(1, 4)], [j for j in range(height - 4, height)]
    # tr_grid_world_rows: [1,2,3], tr_grid_world_cols: [7,8,9,10]
    tr_grid_goal_locs = list(itertools.product(tr_grid_world_rows, tr_grid_world_cols))
    # tr_grid_goal_locs: len(12)
    grid_goal_locs = tl_grid_goal_locs + tr_grid_goal_locs
    # grid_goal_locs: len(28)

    # Four room.
    four_room_goal_locs = [(width, height), (width, 1), (1, height), (1, height - 2), (width - 2, height - 2), (width - 2, 1)]

        # SPREAD vs. TIGHT
    spread_goal_locs = [(width, height), (width, 1), (1, height), (1, height - 2), (width - 2, height - 2), (width - 2, 1), (2,2)]
    tight_goal_locs = [(width, height), (width-1, height), (width, height-1), (width, height - 2), (width - 2, height), (width - 1, height-1)]
    two_room_goal_locs = [(1, height), (3, height), (5, height), (7, height), (9, height), (11,height)]
    # tight_goal_locs = [(6,11),(7,11),(8,11),(9,11),(10,11),(11,11),(7,10),(8,10),(9,10),(10,10),(11,10),
    #                     (8,9),(9,9),(10,9),(11,9),(9,10),(10,10),(11,10),(10,7),(11,7),(11,6)]
    two_room_wall_locs = [(1,6),(2,6),(3,6),(4,6),(5,6),(7,6),(8,6),(9,6),(10,6),(11,6)]
    # maze_goal_locs = [(11,11), (3,10), (10,4), (5,8), (11, 11)]

    changing_entities = {"four_room":four_room_goal_locs,
                    "grid":grid_goal_locs,
                    "corridor":corr_goal_locs,
                    "spread":spread_goal_locs,
                    "tight":tight_goal_locs,
                    # "maze":maze_goal_locs,
                    "chain":[0.0, 0.01, 0.1, 0.5, 1.0],
                    "combo_lock":[[3,1,2],[3,2,1],[2,3,1],[3,3,1]],
                    "maze": make_wall_permutations(mdp_size),
                    "lava": make_lava_permutations(mdp_size),
                    "navigation": navigation_goal_locs,
                    "two_room": two_room_goal_locs
                    }

    # MDP Probability.
    num_mdps = 10 if mdp_class not in list(changing_entities.keys()) else len(changing_entities[mdp_class])
    if mdp_class == "octo":
        num_mdps = 12
    mdp_prob = 1.0 / num_mdps

    for i in range(num_mdps):

        new_mdp = {"chain":ChainMDP(reset_val=changing_entities["chain"][i%len(changing_entities["chain"])]),
                    # "lava":GridWorldMDP(width=width, height=height, rand_init=False, step_cost=-0.001, lava_cost=0.0, lava_locs=changing_entities["lava"][i%len(changing_entities["lava"])], goal_locs=[(mdp_size, mdp_size)], is_goal_terminal=is_goal_terminal, name="lava_world"),
                    "lava":GridWorldMDP(width=width, height=height, rand_init=False, step_cost=0.0, lava_cost=-0.01, lava_locs=changing_entities["lava"][i%len(changing_entities["lava"])], goal_locs=[(mdp_size, mdp_size)], is_goal_terminal=is_goal_terminal, name="lava_world"),
                    "maze":GridWorldMDP(width=width, height=height, rand_init=False, step_cost=0.0, walls=changing_entities["maze"][i%len(changing_entities["maze"])], goal_locs=[(mdp_size, mdp_size)], is_goal_terminal=is_goal_terminal, name="maze_world"),
                    # "maze":GridWorldMDP(width=width, height=height, rand_init=False, step_cost=0.0, walls=maze_walls, goal_locs=[changing_entities["maze"][i%len(changing_entities["maze"])]], is_goal_terminal=is_goal_terminal, name="maze_world"),
                    "four_room":FourRoomMDP(width=width, height=height, rand_init=False, goal_locs=[changing_entities["four_room"][i % len(changing_entities["four_room"])]], is_goal_terminal=is_goal_terminal),
                    "octo":make_grid_world_from_file("octogrid.txt", num_goals=12, randomize=False, goal_num=i),
                    "corridor":GridWorldMDP(width=20, height=1, init_loc=(10, 1), goal_locs=[changing_entities["corridor"][i % len(changing_entities["corridor"])]], is_goal_terminal=is_goal_terminal, name="corridor"),
                    "combo_lock":ComboLockMDP(combo=changing_entities["combo_lock"][i%len(changing_entities["combo_lock"])]),
                    "spread":GridWorldMDP(width=width, height=height, rand_init=False, goal_locs=[changing_entities["spread"][i % len(changing_entities["spread"])]], is_goal_terminal=is_goal_terminal, name="spread_grid"),
                    "tight":GridWorldMDP(width=width, height=height, rand_init=False, goal_locs=[changing_entities["tight"][i % len(changing_entities["tight"])]], is_goal_terminal=is_goal_terminal, name="tight_grid"),
                    "two_room":GridWorldMDP(width=width, height=height, rand_init=False, walls=two_room_wall_locs, goal_locs=[changing_entities["two_room"][i % len(changing_entities["two_room"])]], is_goal_terminal=is_goal_terminal, name="two_room"),
                    }[mdp_class]

        new_mdp.set_gamma(gamma)
        mdp_dist_dict[new_mdp] = mdp_prob

    return MDPDistribution(mdp_dist_dict, horizon=horizon)

def make_wall_permutations(mdp_size):
    # wall_one = [(1,3),(2,3),(3,3),(4,3),(4,2),(6,2),(6,3),(6,4),(7,4),(8,4),(9,4),(10,4),(10,5),(10,6),(10,7),(10,8),(9,10),(10,10),(11,10),(4,7),(5,7),(6,7)]
    # wall_two = [(1,2),(2,2),(3,2),(4,2),(5,2),(6,2),(7,2),(8,2),(9,2),(9,3),(1,4),(2,4),(3,4),(4,4),(5,4),(6,4),(7,4),(8,4),(9,4),(7,7),(8,7),(9,7),(10,7),(11,7),(10,9),(10,10),(10,11)]
    # wall_three = [(1,3),(2,3),(3,3),(4,3),(5,3),(6,3),(7,3),(8,3),(9,3),(5,2),(11,3),(2,6),(3,6),(4,6),(6,6),(7,6),(8,6),(9,6),(10,6),(11,6)]
    # wall_four = [(1,3),(2,3),(3,3),(4,3),(5,3),(6,3),(6,2),(8,2),(8,3),(9,3),(10,3),(11,3),(1,5),(2,5),(3,5),(4,5),(5,5),(6,5),(7,5),(8,5),(4,9),(4,10),(5,9),(6,9),(7,9),(8,9),(9,9),(10,9),(11,9)]
    # wall_five = [(3,2),(3,3),(3,4),(4,4),(5,4),(6,4),(7,4),(6,2),(7,2),(8,2),(9,2),(10,2),(10,3),(10,4),(10,5),(10,6),(10,7),(10,8),(5,8),(6,8),(7,8),(8,8),(9,8),(10,8),(3,8),(3,9),(3,10),(9,10),(10,10),(11,10)]
    # # # # wall_six = [(10,1),(10,2), (10,3), (10,4), (10,5),(10,6), (10,7)]
    # wall_one = [(2,3),(3,2),(3,3),(2,7),(3,6),(4,5),(6,4),(7,5),(9,10),(10,10),(10,11)]
    # wall_two = [(2,2),(1,5),(3,5),(5,5),(7,5),(9,5),(5,8),(4,9),(6,9),(9,10),(10,9)]
    # wall_three = [(1,3),(2,2),(2,6),(4,6),(5,5),(6,5),(7,6),(9,2),(9,3),(10,9),(9,10)]
    # wall_four = [(2,5),(3,4),(4,3),(8,3),(9,4),(10,5),(4,8),(5,7),(6,8),(10,9),(9,10)]
    # wall_five = [(2,3),(3,2),(5,3),(6,2),(8,3),(9,2),(5,6),(7,6),(6,7),(10,10),(11,10)]
    ######
    # wall_one = [(2,3),(3,2),(3,3),(2,7),(3,6),(4,5),(7,5),(8,6),(9,7),(5,8),(7,8),(6,9),(9,10),(10,10),(11,10)]
    # wall_two = [(2,2),(1,5),(3,5),(5,5),(7,5),(9,5),(11,5),(4,4),(10,4),(3,8),(5,8),(4,9),(6,9),(9,10),(10,9)]
    # wall_three = [(1,3),(2,2),(2,6),(4,6),(5,5),(6,5),(8,3),(8,6),(8,10),(9,2),(9,3),(9,5),(10,5),(10,9),(9,10)]
    # wall_four = [(2,5),(3,4),(4,3),(8,3),(9,4),(10,5),(3,7),(4,8),(5,7),(6,8),(7,7),(8,8),(9,7),(10,9),(9,10)]
    # wall_five = [(1,3),(2,2),(4,3),(5,2),(7,3),(8,2),(10,3),(11,2),(4,7),(5,6),(7,6),(6,7),(8,7),(10,10),(11,10)]
    # # wall_six = [(2,1),(2,4),(2,5),(2,9),(3,3),(4,3),(7,3),(7,4),(6,4),(9,4),(10,3),(3,8),(10,9),(10,11),(11,8)]
    ######
    wall_one = coord_from_binary_list(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
    wall_two = coord_from_binary_list(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
    wall_three = coord_from_binary_list(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
    wall_four = coord_from_binary_list(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
    wall_five = coord_from_binary_list(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
    walls = [wall_one, wall_two, wall_three, wall_four, wall_five]
    return walls

def make_lava_permutations(mdp_size):
    # lava_one = [(2,2), (3,3), (4,4), (5,5), (6,6)]
    # lava_two = [(4,1), (1,4), (2,5), (5,2), (7,7)]
    # lava_three = [(3,6), (6,3), (7,7), (9,8), (9,9)]
    # lava_four = [(5,5), (1,8), (8,2), (3,3), (4,3)]
    # lava_five = [(4,1), (2,1), (3,3), (4,3), (5,5), (6,6), (5,7), (4,5)]
    lava_one = [(2,2), (3,3), (4,4), (5,5), (6,6)]
    lava_two = [(2,5), (5,6), (7,7), (8,8), (9,9)]
    lava_three = [(3,4), (4,6), (6,8), (8,10), (10,9)]
    lava_four = [(2,3), (3,5), (5,7), (7,9), (9,10)]
    lava_five = [(4,1), (6,2), (8,4), (9,6), (10,8)]
    lava_six = [(1,4), (2,6), (4,8), (6,9), (8,10)]
    lavas = [lava_one, lava_two, lava_three, lava_four, lava_five, lava_six]
    return lavas
