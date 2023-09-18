import numpy as np


def labels_to_onehot(label):
    onehot = np.zeros(5)
    onehot[int(label)] = 1
    return onehot


def onehot_to_dof(onehot_vector):
    onehot_vector = np.array(onehot_vector, dtype=float)
    label_to_dof = np.array([
        [0, 0],
        [0, -1],  # left
        [-1, 0],  # back
        [0, 1],   # right
        [1, 0],   # front
    ])
    
    dof_cmd = np.dot(onehot_vector, label_to_dof)
    
    norm = np.linalg.norm(dof_cmd)
    if norm > 0:
        dof_cmd /= norm
    else:
        dof_cmd = np.zeros(2)

    return dof_cmd