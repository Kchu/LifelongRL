import os
from simple_rl.utils import chart_utils
from simple_rl.plot_utils import lifelong_plot
from simple_rl.agents.AgentClass import Agent

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
    data_dir = ["D:\\MyPapers\\Results_vs_Tasks\\DelayedQ-FourRoom\\", "D:\\MyPapers\\Results_vs_Tasks\\DelayedQ-Lava\\",
                "D:\\MyPapers\\Results_vs_Tasks\\DelayedQ-Maze\\", "D:\\MyPapers\\Results_vs_Tasks\\Q-FourRoom\\", "D:\\MyPapers\\Results_vs_Tasks\\Q-Lava\\",
                "D:\\MyPapers\\Results_vs_Tasks\\Q-Maze\\"]
    output_dir = "D:\\MyPapers\\Plots\\"

    # Format data dir

    # Grab agents
    
    # Plot.
    for index in range(len(data_dir)):
        print('Plotting ' + str(index+1) +'th figure.')
        agent_names = chart_utils._get_agent_names(data_dir[index])
        agents = []
        actions = []
        if len(agent_names) == 0:
            raise ValueError("Error: no csv files found.")
        for i in agent_names:
            agent = Agent(i, actions)
            agents.append(agent)

        # Grab experiment settings
        episodic = chart_utils._is_episodic(data_dir[index])
        track_disc_reward = chart_utils._is_disc_reward(data_dir[index])
        mdp_name = _get_MDP_name(data_dir[index])
        lifelong_plot(
            agents,
            data_dir[index],
            output_dir,
            n_tasks=40,
            n_episodes=100,
            confidence=0.95,
            open_plot=True,
            plot_title=True,
            plot_legend=True,
            legend_at_bottom=False,
            episodes_moving_average=False,
            episodes_ma_width=10,
            tasks_moving_average=False,
            tasks_ma_width=10,
            latex_rendering=False,
            figure_title=mdp_name)
if __name__ == "__main__":
    main()