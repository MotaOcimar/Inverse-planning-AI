from grid import Node, NodeGrid
from math import inf
import heapq

# Respostas do lab02

class PathPlanner(object):
    """
    Represents a path planner, which may use Dijkstra, Greedy Search or A* to plan a path.
    """
    def __init__(self, cost_map):
        """
        Creates a new path planner for a given cost map.

        :param cost_map: cost used in this path planner.
        :type cost_map: CostMap.
        """
        self.cost_map = cost_map
        self.node_grid = NodeGrid(cost_map)

    @staticmethod
    def construct_path(goal_node):
        """
        Extracts the path after a planning was executed.

        :param goal_node: node of the grid where the goal was found.
        :type goal_node: Node.
        :return: the path as a sequence of (x, y) positions: [(x1,y1),(x2,y2),(x3,y3),...,(xn,yn)].
        :rtype: list of tuples.
        """
        node = goal_node
        # Since we are going from the goal node to the start node following the parents, we
        # are transversing the path in reverse
        reversed_path = []
        while node is not None:
            reversed_path.append(node.get_position())
            node = node.parent
        return reversed_path[::-1]  # This syntax creates the reverse list

    def dijkstra(self, start_position, goal_position):
        """
        Plans a path using the Dijkstra algorithm.

        :param start_position: position where the planning stars as a tuple (x, y).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (x, y).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """

        self.node_grid.reset()
        goal_node = None

        # inicializa a raiz:
        start = self.node_grid.grid[start_position]
        start.g = 0
        start.closed = False  # Open to enter the loop. There it will be closed

        # inicializa heap:
        sorted_nodes = []
        heapq.heappush(sorted_nodes, (start.g, start))

        while len(sorted_nodes) != 0:
            _, actual_node = heapq.heappop(sorted_nodes)

            if not actual_node.closed:
                actual_node.closed = True
                if actual_node.get_position() == goal_position:
                    goal_node = actual_node
                    break

                actual_position = actual_node.get_position()
                for successor_position in self.node_grid.get_successors(actual_position[0], actual_position[1]):
                    successor = self.node_grid.grid[successor_position]
                    cost_actual_node_to_successor = self.cost_map.get_edge_cost(actual_position, successor_position)

                    if successor.g > actual_node.g + cost_actual_node_to_successor:
                        successor.parent = actual_node
                        successor.g = actual_node.g + cost_actual_node_to_successor
                        heapq.heappush(sorted_nodes, (successor.g, successor))

        return self.construct_path(goal_node), goal_node.g

    def greedy(self, start_position, goal_position):
        """
        Plans a path using greedy search.

        :param start_position: position where the planning stars as a tuple (x, y).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (x, y).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """

        self.node_grid.reset()
        goal_node = None

        # inicializa a raiz:
        start = self.node_grid.grid[start_position]
        start.g = 0
        start.h = start.distance_to(goal_position[0], goal_position[1])
        start.closed = False  # aberto para entrar no loop. La sera fechado

        # inicializa heap:
        sorted_nodes = []
        heapq.heappush(sorted_nodes, (start.h, start))

        while len(sorted_nodes) != 0:
            _, actual_node = heapq.heappop(sorted_nodes)

            if not actual_node.closed:
                actual_node.closed = True
                if actual_node.get_position() == goal_position:
                    goal_node = actual_node
                    break

                actual_position = actual_node.get_position()
                for successor_position in self.node_grid.get_successors(actual_position[0], actual_position[1]):
                    successor = self.node_grid.grid[successor_position]
                    if not successor.closed:
                        successor.parent = actual_node
                        successor.h = successor.distance_to(goal_position[0], goal_position[1])
                        cost_actual_node_to_successor = self.cost_map.get_edge_cost(actual_position, successor_position)
                        successor.g = actual_node.g + cost_actual_node_to_successor
                        heapq.heappush(sorted_nodes, (successor.h, successor))

        return self.construct_path(goal_node), goal_node.g

    def a_star(self, start_position, goal_position):
        """
        Plans a path using A*.

        :param start_position: position where the planning stars as a tuple (x, y).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (x, y).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """

        self.node_grid.reset()
        goal_node = None

        # inicializa a raiz:
        start = self.node_grid.grid[start_position]
        start.g = 0
        start.h = start.distance_to(goal_position[0], goal_position[1])
        start.f = start.g + start.h
        start.closed = False  # aberto para entrar no loop. La sera fechado

        # inicializa heap:
        sorted_nodes = []
        heapq.heappush(sorted_nodes, (start.f, start))

        while len(sorted_nodes) != 0:
            _, actual_node = heapq.heappop(sorted_nodes)

            if not actual_node.closed:
                actual_node.closed = True
                if actual_node.get_position() == goal_position:
                    goal_node = actual_node
                    break

                actual_position = actual_node.get_position()
                for successor_position in self.node_grid.get_successors(actual_position[0], actual_position[1]):
                    successor = self.node_grid.grid[successor_position]
                    cost_actual_node_to_successor = self.cost_map.get_edge_cost(actual_position, successor_position)
                    successor.h = successor.distance_to(goal_position[0], goal_position[1])

                    if successor.f > actual_node.g + cost_actual_node_to_successor + successor.h:
                        successor.parent = actual_node
                        successor.g = actual_node.g + cost_actual_node_to_successor
                        successor.f = successor.g + successor.h
                        heapq.heappush(sorted_nodes, (successor.f, successor))

        return self.construct_path(goal_node), goal_node.g
