import numpy as np
import matplotlib.pyplot as plt
from path_planner import PathPlanner
from grid import CostMap
from math import inf
import random
import time

# TODO: Adjust this file to define callable methods that provides the input data of the neural network


class MonteCarlo:
    def __init__(self, width=160, height=120, obstacle_width=20, obstacle_height=15, num_obstacles=20):

        self.width = width
        self.height = height
        self.obstacle_width = obstacle_width
        self.obstacle_height = obstacle_height
        self.num_obstacles = num_obstacles

    def cut_path(self, path, remaining):
        """
        Removes information from the end of the planned path

        :param path: Sequence of (x, y) positions: [(x1,y1),(x2,y2),(x3,y3),...,(xn,yn)].
        :type path: List of tuples.
        :param remaining: How much of the path will remain
        :type remaining: float
        :return: Partial path
        :rtype: List of tuples.
        """
        # TODO
        partial_path = path
        return partial_path

    def write_path_on_map(self, path, cost_map):
        """
        Write the path on the cost map: obstacle (-1), free (1), path (0)
        """
        # TODO
        map_with_path = cost_map
        return map_with_path

    def generate_paths(self, cost_map):
        problem_valid = False

        while not problem_valid:
            # Trying to generate a new problem
            start_position = (random.randint(0, self.height - 1), random.randint(0, self.width - 1))
            goal_position = (random.randint(0, self.height - 1), random.randint(0, self.width - 1))
            # If the start or goal positions happen to be within an obstacle, we discard them and
            # try new samples
            if cost_map.is_occupied(start_position[0], start_position[1]):
                continue
            if cost_map.is_occupied(goal_position[0], goal_position[1]):
                continue
            if start_position == goal_position:
                continue
            problem_valid = True

        path_planner = PathPlanner(cost_map)
        # plt.matshow(cost_map.grid)
        # plt.plot(start_position[1], start_position[0], 'g*', markersize=8)
        # plt.plot(goal_position[1], goal_position[0], 'rx', markersize=8)
        # plt.show()

        dijkstra_path, cost = path_planner.dijkstra(start_position, goal_position)
        greedy_path, cost = path_planner.greedy(start_position, goal_position)
        a_star_path, cost = path_planner.a_star(start_position, goal_position)

        return dijkstra_path, greedy_path, a_star_path

    def shuffle_maps(self, maps, goals):
        # TODO
        shuffled_maps = maps
        shuffled_goals = goals
        return shuffled_maps, shuffled_goals

    def generate_data(self, num_iterations, remaining):
        """
        Generate the paths for each path_planner and adjust data to be a input to the neural network

        :param num_iterations: number of different maps to be created
        :type num_iterations: Integer
        :param remaining: How much of the path will remain
        :type remaining: float
        :return: Lists of maps with the obstacles, partial paths and the goal
        :rtype: List of numpy matrices (with width and height as provided in the constructor) and the list of goals
        """
        maps = []
        goals = []
        for i in range(num_iterations):
            cost_map = CostMap(self.width, self.height)
            random.seed(i)
            cost_map.create_random_map(self.obstacle_width, self.obstacle_height, self.num_obstacles)

            dijkstra_path, greedy_path, a_star_path = self.generate_paths(cost_map)

            goals.append(dijkstra_path[-1])
            goals.append(greedy_path[-1])
            goals.append(a_star_path[-1])

            dijkstra_partial_path = self.cut_path(dijkstra_path, remaining)
            greedy_partial_path = self.cut_path(greedy_path, remaining)
            a_star_partial_path = self.cut_path(a_star_path, remaining)

            # maps append must maintain the same order of goals append
            maps.append(self.write_path_on_map(dijkstra_partial_path, cost_map))
            maps.append(self.write_path_on_map(greedy_partial_path, cost_map))
            maps.append(self.write_path_on_map(a_star_partial_path, cost_map))

        maps, goals = self.shuffle_maps(maps, goals)

        return maps, goals

'''
# TODO: Relocate/reuse the codes below:
# Daqui para baixo, codigo do Manga

# Select planning algorithm
# algorithm = 'dijkstra'
algorithm = 'greedy'
# algorithm = 'a_star'

# Number of path plannings used in the Monte Carlo analysis
num_iterations = 1
# num_iterations = 10
# num_iterations = 100  # Monte Carlo

# Plot options
save_fig = False  # if the figure will be used to the hard disk
show_fig = False  # if the figure will be shown in the screen
fig_format = 'png'
# Recommended figure formats: .eps for Latex/Linux, .svg for MS Office, and .png for easy visualization in Windows.
# The quality of .eps and .svg is far superior since these are vector graphics formats.


def plot_path(cost_map, start, goal, path, filename, save_fig=True, show_fig=True, fig_format='png'):
    """
    Plots the path.

    :param cost_map: cost map.
    :param start: start position.
    :param goal: goal position.
    :param path: path obtained by the path planning algorithm.
    :param filename: filename used for saving the plot figure.
    :param save_fig: if the figure will be saved to the hard disk.
    :param show_fig: if the figure will be shown in the screen.
    :param fig_format: the format used to save the figure.
    """
    plt.matshow(cost_map.grid)
    x = []
    y = []
    for point in path:
        x.append(point[1])
        y.append(point[0])
    plt.plot(x, y, linewidth=2)
    plt.plot(start[1], start[0], 'y*', markersize=8)
    plt.plot(goal[1], goal[0], 'rx', markersize=8)

    plt.xlabel('x / j')
    plt.ylabel('y / i')
    if 'dijkstra' in filename:
        plt.title('Dijkstra')
    elif 'greedy' in filename:
        plt.title('Greedy Best-First')
    else:
        plt.title('A*')

    if save_fig:
        plt.savefig('%s.%s' % (filename, fig_format), format=fig_format)

    if show_fig:
        plt.show()



# These vectors will hold the computation time and path cost for each iteration,
# so we may compute mean and standard deviation statistics in the Monte Carlo analysis.
times = np.zeros((num_iterations, 1))
costs = np.zeros((num_iterations, 1))
for i in range(num_iterations):
    problem_valid = False
    while not problem_valid:
        # Trying to generate a new problem
        start_position = (random.randint(0, self.height - 1), random.randint(0, self.width - 1))
        goal_position = (random.randint(0, self.height - 1), random.randint(0, self.width - 1))
        # If the start or goal positions happen to be within an obstacle, we discard them and
        # try new samples
        if cost_map.is_occupied(start_position[0], start_position[1]):
            continue
        if cost_map.is_occupied(goal_position[0], goal_position[1]):
            continue
        if start_position == goal_position:
            continue
        problem_valid = True
    tic = time.time()
    if algorithm == 'dijkstra':
        path, cost = path_planner.dijkstra(start_position, goal_position)
    elif algorithm == 'greedy':
        path, cost = path_planner.greedy(start_position, goal_position)
    else:
        path, cost = path_planner.a_star(start_position, goal_position)
    # if path is not None and len(path) > 0:
    path_found = True
    toc = time.time()
    times[i] = toc - tic
    costs[i] = cost
    plot_path(cost_map, start_position, goal_position, path, '%s_%d' % (algorithm, i), save_fig, show_fig, fig_format)


# Print Monte Carlo statistics
print(r'Compute time: mean: {0}, std: {1}'.format(np.mean(times), np.std(times)))
if not (inf in costs):
    print(r'Cost: mean: {0}, std: {1}'.format(np.mean(costs), np.std(costs)))
'''