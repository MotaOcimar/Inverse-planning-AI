import numpy as np
import matplotlib.pyplot as plt
from path_planner import PathPlanner
from grid import CostMap
import random
from sklearn.utils import shuffle


class DataGenerator ():
    def __init__(self, width=64, height=64, obstacle_width=10, obstacle_height=7, num_obstacles=10):

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

    def write_path_on_map(self, path, map_to_overwrite):
        """
        Write the path on the cost map: obstacle (-1), free (1), path (0)
        """
        for point in path:
            map_to_overwrite[point[0]][point[1]] = 0

        return map_to_overwrite

    def write_alternatives_on_map(self, original_goal, cost_map, num_alternatives):
        original_goal_alternative = random.randint(0, num_alternatives-1)
        alternatives_list = []
        map_with_alternatives = cost_map.grid.copy()

        for i in range(num_alternatives):

            if i == original_goal_alternative:
                alternatives_list.append(1)  # '1' means the goal alternative
                map_with_alternatives[original_goal[0]][original_goal[1]] = (i+1)*10
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

                alternatives_list.append(0)  # '0' means there isn't the goal alternative
                map_with_alternatives[goal_position[0]][goal_position[1]] = (i+1)*10

        return map_with_alternatives, alternatives_list

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
                # print(start_position, goal_position)
                # plt.matshow(cost_map.grid)
                # plt.plot(start_position[1], start_position[0], 'g*', markersize=8)
                # plt.plot(goal_position[1], goal_position[0], 'rx', markersize=8)
                # title = str(start_position) + ", " + str(goal_position)
                # plt.title(title)
                # plt.show()

                path_planner = PathPlanner(cost_map)
                dijkstra_path, cost = path_planner.dijkstra(start_position, goal_position)
                greedy_path, cost = path_planner.greedy(start_position, goal_position)
                a_star_path, cost = path_planner.a_star(start_position, goal_position)

                problem_valid = True

            except AttributeError:
                # In case there is no valid path
                continue

        return [dijkstra_path, greedy_path, a_star_path]

    def generate_random_goals(self, num_goals, cost_map):

        random_goals = []
        for i in range(num_goals):
            goal_valid = False
            while not goal_valid:
                goal_position = (random.randint(0, self.height - 1), random.randint(0, self.width - 1))
                if cost_map.is_occupied(goal_position[0], goal_position[1]):
                    continue
            random_goals.append(goal_position)

        return random_goals


    def write_fixed_goals_on_map(self, original_goal, possible_goals, cost_map):
        
        alternatives_list = []
        map_with_alternatives = cost_map.grid.copy()

        for i in range(len(possible_goals)):
            goal = possible_goals[i]
            if goal == original_goal:
                alternatives_list.append(1)  # '1' means the goal alternative
                map_with_alternatives[original_goal[0]][original_goal[1]] = (i+1)*10
            else:
                alternatives_list.append(0)  # '0' means there isn't the goal alternative
                map_with_alternatives[goal[0]][goal[1]] = (i+1)*10

        return map_with_alternatives, alternatives_list

    def generate_paths_with_fixed_goal(self, cost_map, goal_position):
        problem_valid = False

        while not problem_valid:
            # Trying to generate a new problem
            start_position = (random.randint(0, self.height - 1), random.randint(0, self.width - 1))
            # If the start or goal positions happen to be within an obstacle, we discard them and
            # try new samples
            if cost_map.is_occupied(start_position[0], start_position[1]):
                continue
            if start_position == goal_position:
                continue
            try:
                # print(start_position, goal_position)
                # plt.matshow(cost_map.grid)
                # plt.plot(start_position[1], start_position[0], 'g*', markersize=8)
                # plt.plot(goal_position[1], goal_position[0], 'rx', markersize=8)
                # title = str(start_position) + ", " + str(goal_position)
                # plt.title(title)
                # plt.show()

                path_planner = PathPlanner(cost_map)
                dijkstra_path, cost = path_planner.dijkstra(start_position, goal_position)
                greedy_path, cost = path_planner.greedy(start_position, goal_position)
                a_star_path, cost = path_planner.a_star(start_position, goal_position)

                problem_valid = True

            except AttributeError:
                # In case there is no valid path
                continue

        return [dijkstra_path, greedy_path, a_star_path]

    def generate_data(self, num_iterations, remaining, num_alternatives, one_map=False):
        """
        Generate the paths for each path_planner and adjust data to be a input to the neural network

        :param num_alternatives: Number of alternatives of goals (1 correct and num_alternatives-1 fake goals)
        :type num_alternatives: Integer
        :param num_iterations: number of different maps to be created
        :type num_iterations: Integer
        :param remaining: How much of the path will remain
        :type remaining: float
        :param one_map: If True, just one map will be created
        :type one_map: Bool
        :return: Lists of maps with the obstacles, partial paths and alternatives for goal, And list with goals
        :rtype: List of numpy matrices (with width and height as provided in the constructor) and the list of goals
        """
        maps = []
        goals = []

        cost_map = CostMap(self.width, self.height)
        cost_map.create_random_map(self.obstacle_width, self.obstacle_height, self.num_obstacles)

        for i in range(num_iterations):
            if not one_map and i != 0:
                cost_map = CostMap(self.width, self.height)
                cost_map.create_random_map(self.obstacle_width, self.obstacle_height, self.num_obstacles)

            paths = self.generate_paths(cost_map)

            for path in paths:
                planner_map, planner_goal = self.write_alternatives_on_map(path[-1], cost_map, num_alternatives)

                planner_partial_path = self.cut_path(path, remaining)

                planner_map = self.write_path_on_map(planner_partial_path, planner_map)

                goals.append(planner_goal)
                maps.append(planner_map)

        maps, goals = shuffle(maps, goals, random_state=0)

        return maps, goals

    def generate_data_with_fixed_goals(self, num_iterations, remaining, num_alternatives):
        """
        Generate the paths for each path_planner and adjust data to be a input to the neural network

        :param num_alternatives: Number of alternatives of goals (1 correct and num_alternatives-1 fake goals)
        :type num_alternatives: Integer
        :param num_iterations: number of different maps to be created
        :type num_iterations: Integer
        :param remaining: How much of the path will remain
        :type remaining: float
        :param one_map: If True, just one map will be created
        :type one_map: Bool
        :return: Lists of maps with the obstacles, partial paths and alternatives for goal, And list with goals
        :rtype: List of numpy matrices (with width and height as provided in the constructor) and the list of goals
        """
        maps = []
        num_goals = num_alternatives

        cost_map = CostMap(self.width, self.height)
        cost_map.create_random_map(self.obstacle_width, self.obstacle_height, self.num_obstacles)
        

        for i in range(num_iterations):

            goal = random.choice(possible_goals) #goal as pair of coordenates
            paths = self.generate_paths_with_fixed_goal(cost_map, goal)
            for path in paths:
                planner_map, planner_goal = self.write_fixed_goals_on_map(path[-1], cost_map, num_alternatives)
                #  planner_goal is a variable that return goals as a vector of alternatives e.g. [1 0 0 0].
                #  
                planner_partial_path = self.cut_path(path, remaining)

                planner_map = self.write_path_on_map(planner_partial_path, planner_map)

                goals.append(planner_goal)
                maps.append(planner_map)

        maps, goals = shuffle(maps, goals, random_state=0)

        return maps, goals
