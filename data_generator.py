import numpy as np
import matplotlib.pyplot as plt
from path_planner import PathPlanner
from grid import CostMap
import random
from sklearn.utils import shuffle


class DataGenerator:
    def __init__(self, width=64, height=64, obstacle_width=10, obstacle_height=8, num_obstacles=20,
                 one_map=False, static_alternatives=False):

        self.width = width
        self.height = height
        self.obstacle_width = obstacle_width
        self.obstacle_height = obstacle_height
        self.num_obstacles = num_obstacles
        self.one_map = one_map
        self.static_alternatives = static_alternatives

        self.possible_goals = []

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

    def write_path_on_map(self, path, map_to_overwrite):
        """
        Write the path on the cost map: obstacle (-1), free (1), path (-3)
        """
        for point in path:
            map_to_overwrite[point[0]][point[1]] = -3

        return map_to_overwrite

    def write_alternatives_on_map(self, original_goal, cost_map, num_alternatives):

        map_with_alternatives = cost_map.grid.copy()
        for i in range(num_alternatives):
            point = self.possible_goals[i]
            map_with_alternatives[point[0]][point[1]] = (i + 1) * 10

        alternatives_list = np.zeros(num_alternatives)
        original_goal_index = self.possible_goals.index(original_goal)
        alternatives_list[original_goal_index] = 1

        return map_with_alternatives, alternatives_list

    def generate_goals(self, cost_map, num_alternatives):
        self.possible_goals = []
        for i in range(num_alternatives):
            valid_goal = False

            while not valid_goal:
                # Trying to generate a new goal
                goal_position = (random.randint(0, self.height - 1), random.randint(0, self.width - 1))
                # If the goal positions happen to be within an obstacle, we discard it and
                # try new sample
                if cost_map.is_occupied(goal_position[0], goal_position[1]):
                    continue
                valid_goal = True

            self.possible_goals.append(goal_position)

    def generate_paths(self, cost_map):
        problem_valid = False

        # Choose a goal position
        num_alternatives = len(self.possible_goals)
        goal_position = self.possible_goals[random.randint(0, num_alternatives-1)]

        while not problem_valid:
            # Trying to generate a new problem
            start_position = (random.randint(0, self.height - 1), random.randint(0, self.width - 1))
            # If the start happen to be within an obstacle, we discard them and
            # try new samples
            if cost_map.is_occupied(start_position[0], start_position[1]):
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

    def plot_map(self, planner_map, planner_goal):
        map_to_plot = planner_map.copy()
        for i in range(len(planner_map)):
            for j in range(len(planner_map[0])):
                if planner_map[i][j] % 10 == 0 and planner_map[i][j] != 0:
                    alternative_index = int(planner_map[i][j] // 10) - 1
                    map_to_plot[i][j] = alternative_index + 4
                    # if is the true goal
                    if planner_goal[alternative_index] == 1:
                        goal_position = (i, j)

        plt.matshow(map_to_plot)
        plt.plot(goal_position[1], goal_position[0], 'rx', markersize=8)
        plt.show()

    def generate_data(self, num_iterations, remaining, num_alternatives):
        maps = []
        goals = []

        # Generate the cost_map
        cost_map = CostMap(self.width, self.height)
        cost_map.create_random_map(self.obstacle_width, self.obstacle_height, self.num_obstacles)

        # Generate the possible goals (alternatives)
        self.generate_goals(cost_map, num_alternatives)

        for i in range(num_iterations):
            if i != 0:
                if not self.one_map:
                    # Update cost_map and possible goals (alternatives)
                    cost_map = CostMap(self.width, self.height)
                    cost_map.create_random_map(self.obstacle_width, self.obstacle_height, self.num_obstacles)
                    self.generate_goals(cost_map, num_alternatives)
                elif not self.static_alternatives:
                    # Update just the possible goals (alternatives)
                    self.generate_goals(cost_map, num_alternatives)

            paths = self.generate_paths(cost_map)

            for path in paths:
                planner_map, planner_goal = self.write_alternatives_on_map(path[-1], cost_map, num_alternatives)

                planner_partial_path = self.cut_path(path, remaining)

                planner_map = self.write_path_on_map(planner_partial_path, planner_map)

                # self.plot_map(planner_map, planner_goal)

                goals.append(planner_goal)
                maps.append(planner_map)

        maps, goals = shuffle(maps, goals, random_state=0)

        return maps, goals
