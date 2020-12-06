import argparse
import numpy as np
import torch

from config import MODEL_PROTEIN
from utils import fetch_protein, DDPGAgent
from utils import calc_free_energy_score as calc_fe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", "-s",
        type=int,
        default=0,
        help="Random seed. (default=0)")
    
    parser.add_argument("--n-episodes", "-e",
        type=int,
        default=250,
        help="Number of episodes to train agent for. (default=250)")
    
    parser.add_argument("--n-timesteps", "-t",
        type=int,
        default=1000,
        help="Number of timesteps to run each episode for. (default=1000)")

    parser.add_argument("--exploit-freq", "-f",
        type=int,
        default=5,
        help="The agent chooses the exploration action every this number of timesteps. (default=5)")
    
    return parser.parse_args()


def get_next_state(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    next_state = state + action
    next_state = np.where(next_state < -180., 180. - (-next_state - -180.), next_state)
    next_state = np.where(next_state > 180., -180. + (next_state - 180.), next_state)
    return next_state


def ddpg(n_episodes: int, n_timesteps: int, exploit_freq: int, goal_reward: float, start_state: np.ndarray):
    agent = DDPGAgent(start_state.shape[0], start_state.shape[0], (-180, 180))
    eps_rewards = np.array([])
    eps_did_term_early = np.array([])

    for e in range(n_episodes):
        state_dict = {"state": start_state}
        eps_reward = 0
        for t in range(n_timesteps):
            action = agent.get_action(state_dict["state"], t % exploit_freq != 0)
            next_state = get_next_state(state_dict["state"], action)
            reward = -calc_fe(next_state)
            eps_reward += reward

            if np.abs(reward - goal_reward) <= 0.001:
                eps_did_term_early = np.append(eps_did_term_early, True)
                break

            agent.store_transition(state_dict["state"], action, reward, next_state)
            agent.learn()

            state_dict["state"] = next_state

        eps_did_term_early = np.append(eps_did_term_early, False)
        eps_rewards = np.append(eps_rewards, eps_reward)
        np.save("ddpg_results", np.vstack((eps_rewards, eps_did_term_early)))
        print("Episode", e + 1, "- Episodic Reward:", eps_reward, "- Did Episode Finish?", eps_did_term_early[-1])


def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    _, goal_state = fetch_protein(MODEL_PROTEIN)
    start_state = np.full(goal_state.flatten().shape, 180.)

    ddpg(args.n_episodes, args.n_timesteps, args.exploit_freq, -calc_fe(goal_state), start_state)


if __name__ == "__main__":
    main()
