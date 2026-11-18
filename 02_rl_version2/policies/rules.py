import numpy as np

def rule_based(obs):
    # Allocate to the largest weighted potential improvement; fixed heuristic.
    q = np.asarray(obs[:3])
    potential = np.array([0.4, 0.3, 0.3]) * (1 - q)
    return int(np.argmax(potential)) + 1
