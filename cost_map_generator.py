# File which defines the CostMap that will be used in the experiment, which is suposed to be always the same.
# The basica idea is that the origins are the only thing changing and the map and possible
# goals are always the same.

import random
from grid import CostMap
import matplotlib.pyplot as plt

def generate_random_goals(num_goals, cost_map):

    random_goals = []
    for i in range(num_goals):
        goal_valid = False
        while not goal_valid:
            goal_position = (random.randint(0, cost_map.height - 1), random.randint(0, cost_map.width - 1))
            if cost_map.is_occupied(goal_position[0], goal_position[1]):
                continue
            else:
                goal_valid = True
        random_goals.append(goal_position)

    return random_goals

class CostMapGenerator():
    def __init__(self, num_goals, show_map = False):
        # important constants:    
        random.seed(1)
        width, height = 64, 64
        obstacle_width, obstacle_height, num_obstacles = 10, 7, 10        
        
        self.num_goals = num_goals
        self.cost_map = CostMap(width, height)
        self.cost_map.create_random_map(obstacle_width, obstacle_height, num_obstacles)
        self.possible_goals = generate_random_goals(self.num_goals,self.cost_map)
        if show_map:
            plot_path(cost_map=self.cost_map)



def plot_path(cost_map, start=None, goal=None, path = [], filename='filename', save_fig=False, show_fig=True, fig_format='png'):
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
    if x!=[] and y!=[]:
        plt.plot(x, y, linewidth=2)
    if start!=None:
        plt.plot(start[1], start[0], 'y*', markersize=8)
    if goal!= None:
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

if __name__ == "__main__":
    cost_map_generator = CostMapGenerator()
    plot_path(cost_map=cost_map_generator.cost_map)