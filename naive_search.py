import argparse
from typing import Tuple
import numpy as np

from config import MODEL_PROTEIN
from utils import fetch_protein
from utils import squared_distance as sq_dist
from utils import calc_free_energy_score as calc_fe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", "-s",
        type=int,
        default=0,
        help="Random seed. (default=0)")
    
    parser.add_argument("--n-confs-to-explore", "-c",
        type=int,
        default=200,
        help="Number of intermediate conformations to iterate over. (default=100)")
    
    parser.add_argument("--n-angles-to-sample", "-a",
        type=int,
        default=10,
        help="Number of new angles to uniformly sample for each angle of the current conformation. (default=10)")
    
    return parser.parse_args()


def naive_search_iteration(curr_state: np.ndarray, n_angles_to_sample: int) -> Tuple[np.ndarray, float]:
    n_angle_dims, n_residues = curr_state.shape
    assert n_angle_dims == 2

    new_state = curr_state.copy()
    new_fe = calc_fe(new_state)
    for i in range(n_angle_dims):
        for j in range(n_residues):
            sampled_angles = 360 * np.random.random_sample((n_angles_to_sample,)) - 180
            for a in sampled_angles:
                candidate_state = new_state.copy()
                candidate_state[i, j] = a
                candidate_fe = calc_fe(candidate_state)
                if candidate_fe < new_fe:
                    new_state = candidate_state
                    new_fe = candidate_fe

    return new_state, new_fe


def main():
    args = parse_args()
    
    np.random.seed(args.seed)

    _, goal_state = fetch_protein(MODEL_PROTEIN)
    start_state = np.full(goal_state.shape, 180.)

    curr_state = start_state
    fe_scores = np.array([calc_fe(curr_state)])
    sq_dists = np.array([sq_dist(curr_state, goal_state)])

    for _ in range(args.n_confs_to_explore):
        print("Current FE Score:", fe_scores[-1], "| Current Squared Distance to Goal State:", sq_dists[-1])
        
        curr_state, curr_fe = naive_search_iteration(curr_state, args.n_angles_to_sample)
        fe_scores = np.append(fe_scores, curr_fe)
        sq_dists = np.append(sq_dists, sq_dist(curr_state, goal_state))
        
        np.save("naive_search_results", np.vstack((fe_scores, sq_dists)))


if __name__ == "__main__":
    main()
