from typing import Tuple, List
import os
import subprocess
import requests
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
import numpy as np


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


def mse(phi_psi_angles_1: np.ndarray, phi_psi_angles_2: np.ndarray) -> float:
    # calculate mean squared error between the two phi/psi angle matrices
    return ((phi_psi_angles_1 - phi_psi_angles_2) ** 2).mean()
