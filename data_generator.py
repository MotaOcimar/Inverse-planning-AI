import numpy as np
import matplotlib.pyplot as plt
from path_planner import PathPlanner
from grid import CostMap
import random
from sklearn.utils import shuffle


class DataGenerator:
    def __init__(self, width=32, height=32, obstacle_width=5, obstacle_height=4, num_obstacles=20):

        self.width = width
        self.height = height
        self.obstacle_width = obstacle_width
        self.obstacle_height = obstacle_height
        self.num_obstacles = num_obstacles

        self.alternatives_channel = None

    def create_path_channel(self, path, remaining):

        path_channel = - np.ones((self.width, self.height))

        path_size = len(path)
        partial_path_size = int(path_size*remaining)  # Floor

        for i in range(partial_path_size):
            point = path[i]
            path_channel[point[0]][point[1]] = i/path_size  # how much percent did he complete the path

        return path_channel

    def create_alternatives_channel(self, original_goal, cost_map, num_alternatives, use_previous=False):
        original_goal_alternative = random.randint(0, num_alternatives-1)
        alternatives_list = np.zeros(num_alternatives)
        alternatives_list[original_goal_alternative] = 1

        if not use_previous or self.alternatives_channel is None:

            alternatives_channel = - np.ones((self.width, self.height))
            for i in range(num_alternatives):

                if i == original_goal_alternative:
                    alternatives_channel[original_goal[0]][original_goal[1]] = i+1
                else:
                    valid_goal = False

                    while not valid_goal:
                        # Trying to generate a new fake goal
                        goal_position = (random.randint(0, self.height - 1), random.randint(0, self.width - 1))
                        # If the fake goal positions happen to be within an obstacle, we discard it and
                        # try new sample
                        if cost_map.is_occupied(goal_position[0], goal_position[1]):
                            continue
                        valid_goal = True

                    alternatives_channel[goal_position[0]][goal_position[1]] = i+1
                    self.alternatives_channel = alternatives_channel
        else:
            alternatives_channel = self.alternatives_channel

        return alternatives_channel, alternatives_list

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
            try:
                path_planner = PathPlanner(cost_map)
                dijkstra_path, cost = path_planner.dijkstra(start_position, goal_position)
                greedy_path, cost = path_planner.greedy(start_position, goal_position)
                a_star_path, cost = path_planner.a_star(start_position, goal_position)

                problem_valid = True

            except AttributeError:
                # In case there is no valid path
                continue

        # print(start_position, goal_position)
        # plt.matshow(cost_map.grid)
        # plt.plot(start_position[1], start_position[0], 'g*', markersize=8)
        # plt.plot(goal_position[1], goal_position[0], 'rx', markersize=8)
        # title = str(start_position) + ", " + str(goal_position)
        # plt.title(title)
        # plt.show()

        return [dijkstra_path, greedy_path, a_star_path]

    def plot_map(self, planner_map, planner_goal=None):
        plt.matshow(planner_map)

        if planner_goal is not None:
            for i in range(len(planner_map)):
                for j in range(len(planner_map[0])):
                    if planner_map[i][j] != -1:
                        alternative_index = int(planner_map[i][j]) - 1
                        # if is the true goal
                        if planner_goal[alternative_index] == 1:
                            goal_position = (i, j)
                            plt.title(goal_position)
                            plt.plot(goal_position[1], goal_position[0], 'rx', markersize=8)
        plt.show()

    def generate_data(self, num_iterations, remaining, num_alternatives, one_map=False, same_alternatives=False):
        maps = []
        goals = []

        cost_map = CostMap(self.width, self.height)
        cost_map.create_random_map(self.obstacle_width, self.obstacle_height, self.num_obstacles)

        for i in range(num_iterations):
            if not one_map and i != 0:
                cost_map = CostMap(self.width, self.height)
                cost_map.create_random_map(self.obstacle_width, self.obstacle_height, self.num_obstacles)

            obstacle_channel = cost_map.grid

            paths = self.generate_paths(cost_map)
            for path in paths:

                alternatives_channel, planner_goal = self.create_alternatives_channel(path[-1], cost_map,
                                                                                      num_alternatives,
                                                                                      use_previous=same_alternatives)

                path_channel = self.create_path_channel(path, remaining)

                planner_map = np.concatenate((obstacle_channel[..., None], alternatives_channel[..., None],
                                              path_channel[..., None]), axis=2)

                # self.plot_map(obstacle_channel)
                # self.plot_map(path_channel)
                # self.plot_map(alternatives_channel, planner_goal)

                goals.append(planner_goal)
                maps.append(planner_map)

        maps, goals = shuffle(maps, goals, random_state=0)

        return maps, goals
