import sys
import csv
from shutil import copyfile


def csv_path_from_agent(root_path, agent):
    """
    Get the saving path from agent object and root path.
    :param root_path: (str)
    :param agent: (object)
    :return: (str)
    """
    return root_path + '/results-' + agent.get_name() + '.csv'


def lifelong_save(init, path, agent, data=None, instance_number=None):
    """
    Save according to a specific data structure designed for lifelong RL experiments.
    :param init: (bool)
    :param path: (str)
    :param agent: agent object
    :param data: (dictionary)
    :param instance_number: (int)
    :return: None
    """
    full_path = csv_path_from_agent(path, agent)
    if init:
        names = ['instance', 'task', 'episode', 'return', 'discounted_return']
        csv_write(names, full_path, 'w')
    else:
        assert data is not None
        assert instance_number is not None
        n_tasks = len(data['returns_per_tasks'])
        n_episodes = len(data['returns_per_tasks'][0])

        for i in range(n_tasks):
            for j in range(n_episodes):
                row = [str(instance_number), str(i + 1), str(j + 1), data['returns_per_tasks'][i][j],
                       data['discounted_returns_per_tasks'][i][j]]
                csv_write(row, full_path, 'a')


def csv_write(row, path, mode):
    """
    Write a row into a csv.
    :param row: (array-like) written row, array-like whose elements are separated in the output file.
    :param path: (str) path to the edited csv
    :param mode: (str) mode for writing: 'w' override, 'a' append
    :return: None
    """
    with open(path, mode) as csv_file:
        w = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        w.writerow(row)

def save_script(path, script_name='original_script.py'):
    if path[-1] != '/':
        path = path + '/'
    copyfile(sys.argv[0], path + script_name)