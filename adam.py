import numpy as np
from termcolor import colored

from config import MODEL_PROTEIN
from utils import fetch_protein
from utils import calc_free_energy_score as calc_fe
from utils import squared_distance as sq_dist


_, goal_state = fetch_protein(MODEL_PROTEIN)


# ∇_M J, where J is sq_dist
def gradM_sq_dist(M: np.ndarray, M_star: np.ndarray) -> np.ndarray:
    return 2 * M - 2 * M_star

# hyperparameters (taken as recommended by Kingma & Ba [2015])
alpha = 0.05
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

M = np.full(goal_state.shape, 180.)  # initial parameter vector
mt = 0  # 1st moment vector
vt = 0  # 2nd moment vector
t = 0  # timestep

# for logging
sq_dists = np.array([sq_dist(M, M_star := goal_state)])
fe_scores = np.array([calc_fe(M)])
goal_fe = calc_fe(M_star)

# optimizer loop
while np.abs(fe_scores[-1] - goal_fe) > 1e-4:
    t += 1
    gt = gradM_sq_dist(M, M_star)
    mt = beta1 * mt + (1 - beta1) * gt
    vt = beta2 * vt + (1 - beta2) * (gt ** 2)
    m_hatt = mt / (1 - (beta1 ** t))
    v_hatt = vt / (1 - (beta2 ** t))
    M = M - alpha * m_hatt / (np.sqrt(v_hatt) + epsilon)

    sq_dists = np.append(sq_dists, sq_dist(M, M_star))
    fe_scores = np.append(fe_scores, calc_fe(M))
    np.save("adam_results", np.vstack((sq_dists, fe_scores)))
    print("Squared distance to goal state:", sq_dists[-1], "| Current FE score", fe_scores[-1])

print(colored(f"Final conformation\n{M}", "blue"))
print(colored(f"Expected conformation\n{M_star}", "green"))
