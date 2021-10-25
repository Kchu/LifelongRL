#!/usr/bin/env python

###########################################################################################
# Implementation of illustrating results. (Average reward for each episode)
# Author for codes: Chu Kun(kun_chu@outlook.com)
# Reference: https://github.com/Kchu/LifelongRL
###########################################################################################

# Python imports.
import os
from simple_rl.utils import chart_utils

def _get_MDP_name(data_dir):
    '''
    Args:
        data_dir (str)

    Returns:
        (list)
    '''
    try:
        params_file = open(os.path.join(data_dir, "parameters.txt"), "r")
    except IOError:
        # No param file.
        return [agent_file.replace(".csv", "") for agent_file in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, agent_file)) and ".csv" in agent_file]

    MDP_name = []

    for line in params_file.readlines():
        if "lifelong-" in line:
            MDP_name = line.split(" ")[0].strip()
            break

    return MDP_name

def main():
    '''
    Summary:
        For manual plotting.
    '''
    # Parameter
    data_dir = ["D:\\MyPapers\\Results_vs_Episodes\\Q-FourRoom\\", "D:\\MyPapers\\Results_vs_Episodes\\Q-Lava\\",
                "D:\\MyPapers\\Results_vs_Episodes\\Q-Maze\\", "D:\\MyPapers\\Results_vs_Episodes\\DelayedQ-FourRoom\\", 
                "D:\\MyPapers\\Results_vs_Episodes\\DelayedQ-Lava\\",
                "D:\\MyPapers\\Results_vs_Episodes\\DelayedQ-Maze\\"]
    output_dir = "D:\\MyPapers\\Plots\\"

    for index in range(len(data_dir)):
        cumulative = False

        # Format data dir
        # data_dir[index] = ''.join(data_dir[index])
        # print(data_dir[index])
        if data_dir[index][-1] != "\\":
            data_dir[index] = data_dir[index] + "\\"

        # Set output file name
        exp_dir_split_list = data_dir[index].split("\\")
        file_name = output_dir + exp_dir_split_list[-2] + '-Episode.pdf'
        # Grab agents.
        agent_names = chart_utils._get_agent_names(data_dir[index])
        if len(agent_names) == 0:
            raise ValueError("Error: no csv files found.")

        # Grab experiment settings
        episodic = chart_utils._is_episodic(data_dir[index])
        track_disc_reward = chart_utils._is_disc_reward(data_dir[index])
        mdp_name = _get_MDP_name(data_dir[index])
        
        # Plot.
        chart_utils.make_plots(data_dir[index], agent_names, cumulative=cumulative, episodic=episodic, track_disc_reward=track_disc_reward, figure_title=mdp_name, plot_file_name=file_name)

if __name__ == "__main__":
    main()