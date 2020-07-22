import numpy as np
import matplotlib.pyplot as plt
from path_planner import PathPlanner
from grid import CostMap
from math import inf
import random
import time

# TODO: Adjust this file to define callable methods that provides the input data of the neural network
# TODO: Save maps and path in just one matrix


def generate_paths(num_iterations):
    """
    Run the path planners in num_iterations different maps

    :param num_iterations: number of different maps to be created
    :type num_iterations: Integer
    :return: Three lists of paths, each one as a sequence of (x, y) positions: [(x1,y1),(x2,y2),(x3,y3),...,(xn,yn)].
    :rtype: List of list of tuples.
    """
    # TODO
    dijkstra_paths = []
    greedy_paths = []
    a_star_paths = []
    return dijkstra_paths, greedy_paths, a_star_paths


def merge_paths(dijkstra_paths, greedy_paths, a_star_paths):
    """
    Randomly merges the lists of paths to provide an unbiased list of paths

    :param dijkstra_paths: lists of paths, each one as a sequence of (x, y) positions: [(x1,y1),(x2,y2),(x3,y3),...,(xn,yn)].
    :type dijkstra_paths: List of list of tuples
    :param greedy_paths: lists of paths, each one as a sequence of (x, y) positions: [(x1,y1),(x2,y2),(x3,y3),...,(xn,yn)].
    :type greedy_paths: List of list of tuples
    :param a_star_paths: lists of paths, each one as a sequence of (x, y) positions: [(x1,y1),(x2,y2),(x3,y3),...,(xn,yn)].
    :type a_star_paths: List of list of tuples
    :return: a single list of randomly ordered paths
    :rtype: List of list of tuples.
    """
    # TODO
    paths = []
    return paths


def cut_paths(paths, remaining):
    """
    Removes information from the end of the planned path

    :param paths: Lists of paths, each one as a sequence of (x, y) positions: [(x1,y1),(x2,y2),(x3,y3),...,(xn,yn)].
    :type paths: List of list of tuples.
    :param remaining: How much of the path will remain
    :type remaining: float
    :return: Lists of partial paths
    :rtype: List of list of tuples.
    """
    # Todo
    partial_paths = []
    return partial_paths


def generate_input_data(num_iterations, remaining):
    """
    Generate the paths for each path_planner and adjust data to be a input to the neural network

    :param num_iterations: number of different maps to be created
    :type num_iterations: Integer
    :param remaining: How much of the path will remain
    :type remaining: float
    :return: Lists of partial paths and list of goals
    :rtype: List of list of tuples and list of tuples
    """
    dijkstra_paths, greedy_paths, a_star_paths = generate_paths(num_iterations)
    paths = merge_paths(dijkstra_paths, greedy_paths, a_star_paths)
    goals = [x[-1] for x in paths]
    partial_paths = cut_paths(paths, remaining)
    # TODO: Refactor to return the map
    return partial_paths, goals


# TODO: Relocate/reuse the codes below:

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


# Environment's parameters
WIDTH = 160
HEIGHT = 120
OBSTACLE_WIDTH = 20
OBSTACLE_HEIGHT = 15
NUM_OBSTACLES = 20

cost_map = CostMap(WIDTH, HEIGHT)
# Initializing the random seed so we have reproducible results
# Please, do not change the seed
random.seed(15)
# Create a random map
cost_map.create_random_map(OBSTACLE_WIDTH, OBSTACLE_HEIGHT, NUM_OBSTACLES)
# Create the path planner using the cost map
path_planner = PathPlanner(cost_map)
# These vectors will hold the computation time and path cost for each iteration,
# so we may compute mean and standard deviation statistics in the Monte Carlo analysis.
times = np.zeros((num_iterations, 1))
costs = np.zeros((num_iterations, 1))
for i in range(num_iterations):
    problem_valid = False
    while not problem_valid:
        # Trying to generate a new problem
        start_position = (random.randint(0, HEIGHT - 1), random.randint(0, WIDTH - 1))
        goal_position = (random.randint(0, HEIGHT - 1), random.randint(0, WIDTH - 1))
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
