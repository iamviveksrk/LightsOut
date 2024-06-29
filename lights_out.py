import galois
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from IPython.display import clear_output, HTML
from matplotlib import animation
from ipycanvas import Canvas, hold_canvas

GF = galois.GF(2)

class LightsOut:

    def __init__(self, grid, stamp_effect=np.array([(1, 0), (0, 1), (0, 0), (-1, 0), (0, -1)])):
        self.grid = list(map(tuple, (np.array(grid) - np.min(grid, axis=0)).tolist()))
        self.grid.sort(key=lambda i: i[0])
        self.n = len(self.grid)
        self.point_number_map = dict(zip(self.grid, range(self.n)))

        self.k1, self.k2 = np.max(self.grid, axis=0) + 1
        self.cm = ListedColormap(['#E76F51', '#36BA98'])
        self.alphas = np.zeros((self.k1, self.k2))
        for i in self.grid:
            self.alphas[i] = 1

        self.stamp_effect = stamp_effect
        
        self.action_matrix = GF(np.array([self.act_vec(point) for point in self.grid], dtype=int)) 
    
    def point_number(self, point):
        '''Returns the name/number of a particular point/location on the grid'''
        return self.point_number_map[tuple(point)]

    def number_point(self, number):
        '''Opposite of point_number'''
        assert number<self.n
        return self.grid[number]
    
    def stamp(self, point):
        '''Return co-ordinates of affected regions when stamped'''
        return [i for i in (np.array(point) + self.stamp_effect) if tuple(i) in self.grid]
    
    def act_vec(self, point):
        '''Creates the action vector when "stamped" on a particular point'''
        result = np.zeros(self.n)
        for affected in self.stamp(point):
            result[self.point_number(affected)] = 1
        
        return result
    
    def state_to_plane(self, state_vec):
        '''Constructs displayable plane from a state configuration'''
        plane = -1 * np.ones((self.k1, self.k2))
        for i in range(self.n):
            if state_vec[i]:
                plane[self.number_point(i)] = 1
            else:
                plane[self.number_point(i)] = 0
        return plane
    
    def displane(self, plane, ax=None, label=True, config=None):
        
        if ax is None:
            ax = plt.axes()
        
        if config is not None:
            plane = self.state_to_plane(plane)

        plt.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])

        cm = ListedColormap(['#E76F51', '#36BA98'])

        frame = [ax.imshow(plane, alpha=(self.alphas).astype(float), vmin=-1, vmax=2, cmap=cm)]

        if label:
            for (i, j) in self.grid:
                frame.append(ax.text(j, i, self.point_number((i, j)), ha="center", va="center"))
        
        return ax, frame
    
    def play(self, state = None):

        if state is None:
            state = GF(np.ones(self.n).astype(int))

        plane = self.state_to_plane(state)
        self.displane(plane)
        plt.show()

        while any(state):
            next_move = int(input('Enter button label to press: '))
            if next_move == -1:
                print('Giving up...')
                break
            
            if next_move not in range(self.n):
                print('Enter valid label!')
            else:
                state += self.action_matrix[:, next_move]
                
                plane = self.state_to_plane(state)
                self.displane(plane)
                
                clear_output()
                plt.show()
        else:
            print('Solved!')
    
    def solve(self, state = None):

        if state is None:
            state = GF(np.ones(self.n).astype(int))
        
        self.A = np.c_[self.action_matrix, state]
        self.A_rref = self.A.row_reduce()

        reqs = np.arange(self.n)[self.A_rref[:, -1]==1]
        
        moves = []
        for i in reqs:
            for j in range(self.n):
                a = self.A_rref[:, j]
                if a[i] == 1 and not any(np.concatenate((a[:i], a[i+1:]))):
                    # print(a, j)
                    moves.append(j)
                    break
        # print(self.A_rref[:, -1])
        return np.array(moves)
    
    def illustrate_moves(self, moves, state=None, label=False):

        if state is None:
            state = GF(np.ones(self.n).astype(int))

        artists = []
        fig, ax = plt.subplots()

        plane = self.state_to_plane(state)
        ax, frame = self.displane(plane, ax=ax, label=label)
        artists.append(frame)

        for i in moves:

            state += self.action_matrix[:, i]
            plane = self.state_to_plane(state)
            ax, frame = self.displane(plane, ax=ax, label=label)
            artists.append(frame)

        plt.close()
        interval = 500 if label else 50
        anim = animation.ArtistAnimation(fig, artists, interval=interval, repeat=False, blit=False);
        return HTML(anim.to_jshtml())

    
    def illustrate_solution(self, state=None, label=False):
        sol = self.solve(state)
        return self.illustrate_moves(sol, state=state, label=label)
        