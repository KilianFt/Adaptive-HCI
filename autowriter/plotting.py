import numpy as np
import matplotlib.pyplot as plt


def label_to_move(label):
    move_map = [[-1, 0], [0, -1], [1, 0], [0, 1]]

    if label >= len(move_map):
         return [0, 0]

    return move_map[label]


def get_color(k):
    scol = ['r','g','b','m','c']
    ncol = len(scol)
    if k < ncol:
       out = scol[k]
    else:
       out = scol[-1]
    return out


def plot_traj(stk, color, lw, ax):
	n = stk.shape[0]
	if n > 1:
		ax.plot(stk[:,0],stk[:,1],color=color, linewidth=lw)
	else:
		ax.plot(stk[0,0],stk[0,1],color=color, linewidth=lw,marker='.')


def plot_motor_to_image(I, drawing, ax, lw=2):
	drawing = [d[:,0:2] for d in drawing] # strip off the timing data (third column)
	ax.imshow(I,cmap='gray')
	ns = len(drawing)
	for sid in range(ns): # for each stroke
		plot_traj(drawing[sid], get_color(sid), lw, ax)
	ax.set_xticks([])
	ax.set_yticks([])
      

def plot_moves(moves, canvas_size=120, ax=None):
    canvas = np.zeros((canvas_size, canvas_size))
    last_point = np.array([canvas_size // 2, canvas_size // 2])
    canvas[last_point[0], last_point[1]] = 1

    for move in moves:
        point = last_point + move
        canvas[point[0], point[1]] = 1
        last_point = point

    if ax is not None:
        ax.imshow(canvas)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        plt.imshow(canvas)


def plot_encoded_moves(encoded_moves, canvas_size=120, ax=None):
    if not isinstance(encoded_moves, list):
         encoded_moves = encoded_moves.tolist()

    moves = [label_to_move(x) for x in encoded_moves]
    plot_moves(moves, canvas_size=canvas_size, ax=ax)
