#!/usr/bin/env python3
"""
Unified Docking Script for Synthformer Experiments
==================================================

This script combines all docking experiments into a single command-line tool 
with different modes for different experimental setups.

Usage:
    python dock_experiments.py --mode [standard|random|squid|chemprojector|hit_expansion] [options]

Modes:
    standard      - Standard docking using pickle files (dock_generate_molecule.py)
    random        - Random docking using text files (dock_generate_molecule_rand.py)
    squid         - SQUID-specific docking using CSV input (dock_generate_molecule_SQUID.py)
    chemprojector - ChemProjector docking using CSV input (dock_generate_molecule_chemprojector.py)
    hit_expansion - Hit expansion docking (dock_hit_expansion.py)
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
import subprocess
import pickle
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolTransforms import ComputeCentroid

from openbabel import pybel
from opencadd.structure.core import Structure


def pdb_to_pdbqt(pdb_path, pdbqt_path, pH=7.4):
    """
    Convert a PDB file to a PDBQT file needed by docking programs of the AutoDock family.

    Parameters
    ----------
    pdb_path: str or pathlib.Path
        Path to input PDB file.
    pdbqt_path: str or pathlib.path
        Path to output PDBQT file.
    pH: float
        Protonation at given pH.
    """
    molecule = list(pybel.readfile("pdb", str(pdb_path)))[0]
    # add hydrogens at given pH
    molecule.OBMol.CorrectForPH(pH)
    molecule.addh()
    # add partial charges to each atom
    for atom in molecule.atoms:
        atom.OBAtom.GetPartialCharge()
    molecule.write("pdbqt", str(pdbqt_path), overwrite=True)
    return


def smiles_to_pdbqt(smiles, pdbqt_path, pH=7.4):
    """
    Convert a SMILES string to a PDBQT file needed by docking programs of the AutoDock family.

    Parameters
    ----------
    smiles: str
        SMILES string.
    pdbqt_path: str or pathlib.path
        Path to output PDBQT file.
    pH: float
        Protonation at given pH.
    """
    molecule = pybel.readstring("smi", smiles)
    # add hydrogens at given pH
    molecule.OBMol.CorrectForPH(pH)
    molecule.addh()
    # generate 3D coordinates
    molecule.make3D(forcefield="mmff94s", steps=10000)
    # add partial charges to each atom
    for atom in molecule.atoms:
        atom.OBAtom.GetPartialCharge()
    molecule.write("pdbqt", str(pdbqt_path), overwrite=True)
    return


def sdf_to_pdbqt(smi_path, pdbqt_path, pH=7.4):
    """
    Convert an SDF file to a PDBQT file needed by docking programs of the AutoDock family.

    Parameters
    ----------
    smi_path: str
        Path to SDF file.
    pdbqt_path: str or pathlib.path
        Path to output PDBQT file.
    pH: float
        Protonation at given pH.
    """
    mol = pybel.readfile("sdf", smi_path)
    for molecule in mol:
        molecule = molecule
        break
    # add hydrogens at given pH
    molecule.OBMol.CorrectForPH(pH)
    molecule.addh()
    # add partial charges to each atom
    for atom in molecule.atoms:
        atom.OBAtom.GetPartialCharge()
    molecule.write("pdbqt", str(pdbqt_path), overwrite=True)
    return


def run_smina(ligand_path, protein_path, out_path, pocket_center, pocket_size, 
              atom_terms, num_poses=10, exhaustiveness=8):
    """
    Perform docking with Smina.

    Parameters
    ----------
    ligand_path: str or pathlib.Path
        Path to ligand PDBQT file that should be docked.
    protein_path: str or pathlib.Path
        Path to protein PDBQT file that should be docked to.
    out_path: str or pathlib.Path
        Path to which docking poses should be saved, SDF or PDB format.
    pocket_center: iterable of float or int
        Coordinates defining the center of the binding site.
    pocket_size: iterable of float or int
        Lengths of edges defining the binding site.
    atom_terms: str
        Path to atom terms file.
    num_poses: int
        Maximum number of poses to generate.
    exhaustiveness: int
        Accuracy of docking calculations.

    Returns
    -------
    output_text: str
        The output of the Smina calculation.
    """
    output_text = subprocess.check_output(
        [
            "smina",
            "--ligand", str(ligand_path),
            "--receptor", str(protein_path),
            "--out", str(out_path),
            "--center_x", str(pocket_center[0]),
            "--center_y", str(pocket_center[1]),
            "--center_z", str(pocket_center[2]),
            "--size_x", str(pocket_size[0]),
            "--size_y", str(pocket_size[1]),
            "--size_z", str(pocket_size[2]),
            "--num_modes", str(num_poses),
            "--exhaustiveness", str(exhaustiveness),
            "--atom_terms", atom_terms
        ],
        universal_newlines=True,
    )
    return output_text


def minimise_smina(ligand_path, protein_path, out_path, atom_terms):
    """
    Perform score-only minimization with Smina.

    Parameters
    ----------
    ligand_path: str or pathlib.Path
        Path to ligand PDBQT file.
    protein_path: str or pathlib.Path
        Path to protein PDBQT file.
    out_path: str or pathlib.Path
        Path to output file.
    atom_terms: str
        Path to atom terms file.

    Returns
    -------
    output_text: str
        The output of the Smina calculation.
    """
    output_text = subprocess.check_output(
        [
            "smina",
            "--ligand", str(ligand_path),
            "--receptor", str(protein_path),
            "--out", str(out_path),
            "--score_only",
            "--atom_terms", str(atom_terms)
        ],
        universal_newlines=True,
    )
    return output_text


def load_smiles_data(mode, folder_path, csv_file=None, target_index=None):
    """Load SMILES data based on the docking mode."""
    if mode == "standard":
        # Load from pickle file
        filepath = os.path.join(folder_path, "my_dict.pkl")
        with open(filepath, "rb") as file:
            data = pickle.load(file)
        return [smile[-1] for smile in data]
    
    elif mode == "random":
        # Load from text file
        file_path = "smiles_list.txt"
        with open(file_path, "r") as file:
            data = file.readlines()
        return [smile.strip() for smile in data]
    
    elif mode in ["squid", "chemprojector"]:
        # Load from CSV file
        if csv_file is None:
            csv_file = 'molecule_data.csv' if mode == "squid" else 'example.csv'
        df = pd.read_csv(csv_file)
        groups = df.groupby('target')
        target_smiles = {target: group['smiles'].tolist() for target, group in groups}
        
        if target_index is not None:
            target = list(target_smiles.keys())[target_index]
            return target_smiles[target], target
        return target_smiles
    
    elif mode == "hit_expansion":
        # Load from text file in hit_expansion directory
        hit_expansion_dir = os.path.join(folder_path, "hit_expansion")
        if not os.path.exists(hit_expansion_dir):
            return None
        filepath = os.path.join(hit_expansion_dir, "moleculess.txt")
        with open(filepath, "r") as file:
            data = file.readlines()
        return [smile.strip() for smile in data]
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def get_docking_parameters(mode):
    """Get docking parameters based on mode."""
    parameters = {
        "standard": {
            "box_size": [20, 20, 20],
            "num_poses": 10,
            "exhaustiveness": 4,
            "file_prefix": "",
            "mol_suffix": "",
            "do_minimization": True
        },
        "random": {
            "box_size": [20, 20, 20],
            "num_poses": 10,
            "exhaustiveness": 4,
            "file_prefix": "r",
            "mol_suffix": "",
            "do_minimization": False
        },
        "squid": {
            "box_size": [20, 20, 20],
            "num_poses": 10,
            "exhaustiveness": 4,
            "file_prefix": "sq_",
            "mol_suffix": "_sq",
            "do_minimization": False
        },
        "chemprojector": {
            "box_size": [20, 20, 20],
            "num_poses": 10,
            "exhaustiveness": 4,
            "file_prefix": "c",
            "mol_suffix": "c",
            "do_minimization": False
        },
        "hit_expansion": {
            "box_size": [20, 20, 20],
            "num_poses": 10,
            "exhaustiveness": 4,
            "file_prefix": "",
            "mol_suffix": "",
            "do_minimization": False
        }
    }
    return parameters.get(mode, parameters["standard"])


def save_smiles_to_file(smiles_list, filepath, mode):
    """Save SMILES list to file for tracking."""
    filename_map = {
        "squid": "SQUID.txt",
        "chemprojector": "chemprojector.txt"
    }
    
    if mode in filename_map:
        save_path = os.path.join(filepath, filename_map[mode])
        with open(save_path, "w") as file:
            for item in smiles_list:
                file.write(str(item) + "\n")
        print(f"List saved as {filename_map[mode]} at {save_path}")


def dock_molecules(mode, max_folders=17, csv_file=None, start_folder=0, verbose=True):
    """
    Main docking function that handles all different modes.
    
    Parameters:
    - mode: Docking mode ('standard', 'random', 'squid', 'chemprojector', 'hit_expansion')
    - max_folders: Maximum number of folders to process
    - csv_file: CSV file for squid/chemprojector modes
    - start_folder: Starting folder index (for hit_expansion mode)
    - verbose: Whether to print detailed output
    """
    sys.path.insert(0, './../')
    
    thisdir = os.getcwd()
    mypath = os.path.join(thisdir, "refined-set")
    folder_path = mypath
    folders = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
               if os.path.isdir(os.path.join(folder_path, f))]
    
    folders.sort()
    ligands = [os.path.join(folder, f"{folder[-4:]}_ligand.mol2") for folder in folders]
    proteins = [os.path.join(folder, f"{folder[-4:]}_protein.pdb") for folder in folders]
    
    if verbose:
        print(f"Proteins: {proteins[:max_folders]}")
    
    # Get docking parameters for this mode
    params = get_docking_parameters(mode)
    
    # Handle CSV data loading for squid/chemprojector modes
    target_smiles_dict = None
    if mode in ["squid", "chemprojector"]:
        target_smiles_dict = load_smiles_data(mode, None, csv_file)
    
    # Determine folder range
    folder_range = folders[start_folder:max_folders] if mode == "hit_expansion" else folders[:max_folders]
    
    for i, folder in enumerate(folder_range):
        actual_index = i + start_folder if mode == "hit_expansion" else i
        
        if verbose:
            print(f"{actual_index} start preparation {folder}")
        
        try:
            # Load ligand and compute centroid
            lig = Chem.rdmolfiles.MolFromMol2File(ligands[actual_index], sanitize=True)
            if lig is None:
                if verbose:
                    print(f"Failed to load ligand for {folder}")
                continue
                
            centroid = ComputeCentroid(lig.GetConformer())
            centroid = [centroid.x, centroid.y, centroid.z]
            
            # Set up file paths
            if mode == "hit_expansion":
                hit_expansion_dir = os.path.join(folder, "hit_expansion")
                if not os.path.exists(hit_expansion_dir):
                    if verbose:
                        print(f"Hit expansion directory not found for {folder}")
                    continue
                work_dir = hit_expansion_dir
                mol_file = os.path.join(folder, 'mol.pdbqt')
            else:
                work_dir = folder
                mol_suffix = params["mol_suffix"]
                mol_file = os.path.join(folder, f'mol{mol_suffix}.pdbqt')
            
            protein_file = proteins[actual_index] + "qt"
            
            # Convert ligand SDF to PDBQT
            ligand_sdf = ligands[actual_index][:-4] + 'sdf'
            sdf_to_pdbqt(ligand_sdf, mol_file)
            
            if verbose:
                print("start minimising")
                print(f"Protein: {proteins[actual_index]}")
            
            # Convert protein PDB to PDBQT
            pdb_to_pdbqt(proteins[actual_index], protein_file)
            
            if verbose:
                print("pdb converted")
            
            # Perform crystal minimization if required
            if params["do_minimization"]:
                crystal_out = os.path.join(folder, "dock_pose_crystal.sdf")
                atom_terms_file = os.path.join(folder, "mol_int.pdb")
                minimise_smina(mol_file, protein_file, crystal_out, atom_terms_file)
                if verbose:
                    print("Crystal minimization completed")
            
            # Load SMILES data
            if mode in ["squid", "chemprojector"]:
                smiles_list, target = load_smiles_data(mode, folder, csv_file, actual_index)
                save_smiles_to_file(smiles_list, folder, mode)
            else:
                smiles_list = load_smiles_data(mode, folder)
                if smiles_list is None:
                    if verbose:
                        print(f"No SMILES data found for {folder}")
                    continue
            
            if verbose:
                print(f"Loaded {len(smiles_list)} SMILES")
            
            # Process each SMILES
            for idx, smiles in enumerate(smiles_list):
                smiles = smiles.strip()
                
                # Set up molecule-specific file paths
                mol_suffix = params["mol_suffix"]
                smi_path = os.path.join(work_dir, f'{idx}_mol{mol_suffix}.pdbqt')
                
                if verbose:
                    print(f"start smile to pdbqt: {smiles}")
                
                # Convert SMILES to PDBQT
                smiles_to_pdbqt(smiles, smi_path)
                
                # Set up output path
                file_prefix = params["file_prefix"]
                out_path = os.path.join(work_dir, f"{file_prefix}{idx}_docked.sdf")
                atom_terms_file = os.path.join(work_dir, f"{file_prefix}{idx}_docked_mol_inter.pdb")
                
                if verbose:
                    print("start dock")
                
                # Run docking
                run_smina(
                    smi_path, 
                    protein_file, 
                    out_path, 
                    centroid, 
                    params["box_size"],
                    atom_terms_file,
                    params["num_poses"],
                    params["exhaustiveness"]
                )
                
                if verbose:
                    print(f"finish dock: {out_path}")
                    
        except Exception as e:
            if verbose:
                print(f"Fail {folder}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Unified Docking Script for Synthformer Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--mode', 
        required=True,
        choices=['standard', 'random', 'squid', 'chemprojector', 'hit_expansion'],
        help='Docking mode to run'
    )
    
    parser.add_argument(
        '--max-folders',
        type=int,
        default=17,
        help='Maximum number of folders to process (default: 17)'
    )
    
    parser.add_argument(
        '--start-folder',
        type=int,
        default=0,
        help='Starting folder index (useful for hit_expansion mode, default: 0)'
    )
    
    parser.add_argument(
        '--csv-file',
        type=str,
        help='CSV file for squid/chemprojector modes (default: molecule_data.csv for squid, example.csv for chemprojector)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Adjust start folder for hit_expansion mode if not specified
    if args.mode == "hit_expansion" and args.start_folder == 0:
        args.start_folder = 4  # Default from original script
    
    # Run docking
    dock_molecules(
        mode=args.mode,
        max_folders=args.max_folders,
        csv_file=args.csv_file,
        start_folder=args.start_folder,
        verbose=not args.quiet
    )
    
    print(f"Docking completed for mode: {args.mode}")


if __name__ == "__main__":
    main() 