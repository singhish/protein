from typing import Tuple, List
import os
import subprocess
import requests
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
from termcolor import colored


###############################
## General Utility Functions ##
###############################
def fetch_protein(pdb_id: str) -> Tuple[List[str], np.ndarray]:
    # retrieve pdb file from Protein Data Bank
    pdb_file = f"{pdb_id}.pdb"
    pdb_file_path = os.path.join(os.getcwd(), pdb_file)
    protein_url = f"https://files.rcsb.org/download/{pdb_file}"
    req = requests.get(protein_url)
    with open(pdb_file_path, "w") as f:
        f.write(req.text)
    
    # parse pdb file
    structure = PDBParser().get_structure(pdb_id, pdb_file)
    peptides = PPBuilder().build_peptides(structure)[0]
    
    # extract amino acid sequence and phi/psi angles
    aa_sequence = list(peptides.get_sequence())
    phi_psi_angles = np.array(
        list(map(
            lambda x: (180 if not x[0] else np.rad2deg(x[0]),
                       180 if not x[1] else np.rad2deg(x[1])),
            peptides.get_phi_psi_list()))).T
    
    # remove pdb file
    subprocess.check_output(["rm", pdb_file])

    return aa_sequence, phi_psi_angles


def calc_free_energy_score(phi_psi_angles: np.ndarray) -> float:
    # reshape phi/psi angle array to have shape (2, n_residues)
    phi_psi_angles = phi_psi_angles.reshape((2, int(phi_psi_angles.flatten().shape[0] / 2)))

    # dump phi/psi angle array to file for input into redcraft
    rc_in_file = "rc_in.txt"
    with open(rc_in_file, "w") as f:
        f.write(" ".join([" ".join([str(a) for a in phi_psi]) for phi_psi in phi_psi_angles.T]))

    # call redcraft
    output = subprocess.check_output(["redcraft", "molan", "-e", "-d", "RDC_new", "-p", ".", "-m", "2", rc_in_file])

    # extract free energy score
    free_energy_score = float(output.decode("utf-8").split()[-1])

    # remove redcraft input file
    subprocess.check_output(["rm", rc_in_file])

    return free_energy_score


def squared_distance(phi_psi_angles_1: np.ndarray, phi_psi_angles_2: np.ndarray) -> float:
    # calculate squared Frobenius distance between the two phi/psi angle matrices
    return np.linalg.norm(phi_psi_angles_1 - phi_psi_angles_2) ** 2


#####################
## Assets for DDPG ##
#####################
class ReplayBuffer:
    def __init__(self,
                 max_size=1000000,
                 sample_batch_size=100):

        self.max_size: int = max_size
        self.sample_batch_size: int = sample_batch_size

        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.next_states: List[np.ndarray] = []

        self.n_samples_recorded: int = 0
        self.current_size: int = 0
    
    def append(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        
        self.n_samples_recorded += 1
        self.current_size += 1

        if self.current_size > self.max_size:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)

            self.current_size -= 1
        

    def sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idxs = np.random.choice(
            np.arange(min(self.n_samples_recorded, self.max_size)),
            size=min(self.n_samples_recorded, self.sample_batch_size),
            replace=False)

        sampled_states = np.array([self.states[i] for i in idxs])
        sampled_actions = np.array([self.actions[i] for i in idxs])
        sampled_rewards = np.array([self.rewards[i] for i in idxs])
        sampled_next_states = np.array([self.next_states[i] for i in idxs])

        return sampled_states, sampled_actions, sampled_rewards, sampled_next_states


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, action_bounds: Tuple[int, int],
                 epsilon=0.003):

        super(Actor, self).__init__()

        assert action_bounds[0] < action_bounds[1]
        self.action_lb = action_bounds[0]
        self.action_ub = action_bounds[1]

        # each hidden layer is initialized using the fan-in process mentioned in the paper
        # this same fan-in process is provided in PyTorch by way of Glorot & Bengio (2010)
        self.fc1 = nn.Linear(state_dim, 400)
        nn.init.xavier_uniform_(self.fc1.weight)

        self.fc2 = nn.Linear(400, 300)
        nn.init.xavier_uniform_(self.fc2.weight)

        self.fc3 = nn.Linear(300, action_dim)
        nn.init.uniform_(self.fc3.weight, -epsilon, epsilon)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # feed through first layer
        x = self.fc1(state)
        x = torch.relu(x)

        # feed through second layer
        x = self.fc2(x)
        x = torch.relu(x)

        # feed through final layer using tanh
        x = self.fc3(x)
        x = torch.tanh(x)

        # scale output to match action bounds
        x = 0.5 * ((self.action_ub - self.action_lb) * x + (abs(self.action_ub) - abs(self.action_lb)))

        return x


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int,
                 epsilon=0.003):

        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim, 400)
        nn.init.xavier_uniform_(self.fc1.weight)

        self.fc2 = nn.Linear(400 + action_dim, 300)
        nn.init.xavier_uniform_(self.fc2.weight)

        self.fc3 = nn.Linear(300, 1)
        nn.init.uniform_(self.fc3.weight, -epsilon, epsilon)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # feed through first layer
        x = self.fc1(state)
        x = torch.relu(x)

        # feed in action at second layer
        x = torch.cat((x, action), dim=1)
        x = self.fc2(x)
        x = torch.relu(x)

        # feed through output layer
        x = self.fc3(x)

        return x


class DDPGAgent:
    def __init__(self, state_dim: int, action_dim: int, action_bounds: Tuple[int, int],
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 gamma=0.99):
        
        assert action_bounds[0] < action_bounds[1]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            print(colored("DDPG agent is utilizing CUDA on your", "green"),
                  colored(torch.cuda.get_device_name(torch.cuda.current_device()), "magenta"))

        self.actor = Actor(state_dim, action_dim, action_bounds).to(self.device)
        self.actor_t = deepcopy(self.actor).to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_t = deepcopy(self.critic).to(self.device)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.action_lb = action_bounds[0]
        self.action_ub = action_bounds[1]

        self.gamma = gamma

        self.replay_buffer = ReplayBuffer()
    
    def _update_params(self,
                       tau=1e-3):

        for param_t, param in zip(self.actor_t.parameters(), self.actor.parameters()):
            param_t.data.copy_(tau * param.data + (1 - tau) * param_t.data)
        
        for param_t, param in zip(self.critic_t.parameters(), self.critic.parameters()):
            param_t.data.copy_(tau * param.data + (1 - tau) * param_t.data)

    def get_action(self, state: np.ndarray,
                   exploit=False) -> np.ndarray:
        """`state` must be passed in as a flattened version of a phi/psi angle matrix
            (e.g. using `state.flatten()`)"""

        state = torch.from_numpy(state).float().to(self.device)
        action = self.actor(state)
        
        if not exploit:
            # using Gaussian noise instead of Ornstein-Uhlenbeck noise
            action += torch.empty(action.size()).normal_().to(self.device)
        
        return action.cpu().detach().numpy()
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, next_state: np.ndarray):
        self.replay_buffer.append(state, action, reward, next_state)
    
    def learn(self):
        states, actions, rewards, next_states = self.replay_buffer.sample()

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)

        # optimize critic
        next_actions = self.actor_t(next_states).detach()
        next_q = torch.squeeze(self.critic_t(next_states, next_actions).detach())
        y_exp = rewards + self.gamma * next_q
        y_pred = torch.squeeze(self.critic(states, actions))
        critic_loss = F.mse_loss(y_pred.squeeze(), y_exp.squeeze())
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # optimize actor
        pred_actions = self.actor(states)
        actor_loss = -torch.sum(self.critic(states, pred_actions))
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self._update_params()
