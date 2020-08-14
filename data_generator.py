import numpy as np
import matplotlib.pyplot as plt
from path_planner import PathPlanner
from grid import CostMap
import random
from sklearn.utils import shuffle


class DataGenerator:
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
        path_size = len(path)
        partial_path_size = int(path_size*remaining)  # Floor
        partial_path = path[:partial_path_size]
        return partial_path

    def write_path_on_map(self, path, cost_map):
        """
        Write the path on the cost map: obstacle (-1), free (1), path (0)
        """
        map_with_path = cost_map.grid.copy()

        for point in path:
            map_with_path[point[0]][point[1]] = 0

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

        dijkstra_path, cost = path_planner.dijkstra(start_position, goal_position)
        greedy_path, cost = path_planner.greedy(start_position, goal_position)
        a_star_path, cost = path_planner.a_star(start_position, goal_position)

        return dijkstra_path, greedy_path, a_star_path

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

        maps, goals = shuffle(maps, goals, random_state=0)

        return maps, goals
