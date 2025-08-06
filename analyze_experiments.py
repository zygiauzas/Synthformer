#!/usr/bin/env python3
"""
Unified Analysis Script for Synthformer Experiments
===================================================

This script combines all docking and molecular analysis experiments into a single
command-line tool with different modes for different experimental setups.

Usage:
    python analyze_experiments.py --mode [docking|docking_txt|chem|squid|hit_expansion] [options]

Modes:
    docking       - Standard docking analysis using pickle files (original analyse_docking.py)
    docking_txt   - Docking analysis using text files (analyse_docking copy.py)
    chem          - Chemistry-focused analysis using CSV input (analyse_docking_chem.py)
    squid         - Squid-specific analysis using CSV input (analyse_docking_Squid.py)
    hit_expansion - Hit expansion analysis (analyse_hit_expansion.py)
"""

import argparse
import matplotlib.pyplot as plt
import os
import sys
import warnings
from pathlib import Path
import subprocess
import pickle
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, Crippen, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem.rdMolTransforms import ComputeCentroid

from openbabel import pybel
from opencadd.structure.core import Structure


def calculate_molecule_properties(smiles_query, smiles_list):
    """
    Calculate molecular properties for a query molecule and a list of molecules.
    
    Properties include molecular weight, logP, drug-likeness (QED), and synthetic accessibility (SA) scores.

    Parameters:
    - smiles_query: SMILES string of the query molecule.
    - smiles_list: List of SMILES strings of the target molecules.
    
    Returns:
    - query_properties: Dictionary with properties of the query molecule.
    - mean_properties: Dictionary with mean properties of the target molecules.
    """
    # Convert the query molecule to an RDKit molecule object
    query_mol = Chem.MolFromSmiles(smiles_query)
    
    # Calculate properties for the query molecule
    query_properties = {
        'Ref Molecular Weight': Descriptors.MolWt(query_mol),
        'Ref LogP': Crippen.MolLogP(query_mol),
        'Ref Drug-Likeness (QED)': QED.qed(query_mol),
    }
    
    # Initialize lists to store properties of the target molecules
    molecular_weights = []
    logp_values = []
    drug_likeness_values = []

    # Loop over the list of target molecules and calculate their properties
    for smiles in smiles_list:
        target_mol = Chem.MolFromSmiles(smiles)
        if target_mol is not None:
            molecular_weights.append(Descriptors.MolWt(target_mol))
            logp_values.append(Crippen.MolLogP(target_mol))
            drug_likeness_values.append(QED.qed(target_mol))
    
    # Calculate mean and variance of the properties for the target molecules
    mean_properties = {
        'Gen mean Molecular Weight': round(np.mean(molecular_weights), 2),
        'Molecular Weight 5th Percentile': round(np.percentile(molecular_weights, 5), 2),
        'Molecular Weight 95th Percentile': round(np.percentile(molecular_weights, 95), 2),
        'Gen mean LogP': round(np.mean(logp_values), 2),
        'LogP 5th Percentile': round(np.percentile(logp_values, 5), 2),
        'LogP 95th Percentile': round(np.percentile(logp_values, 95), 2),
        'Gen mean Drug-Likeness (QED)': round(np.mean(drug_likeness_values), 2),
        'Drug-Likeness 5th Percentile': round(np.percentile(drug_likeness_values, 5), 2),
        'Drug-Likeness 95th Percentile': round(np.percentile(drug_likeness_values, 95), 2),
    }

    return query_properties, mean_properties


def calculate_all_similarities(smiles_query, smiles_list, fingerprint_bits=1024, sanitize_mols=False):
    """
    Calculate Tanimoto, Murcko scaffold-based, and Gobbi (MACCS keys) similarities 
    between a query molecule and a list of molecules.
    
    Parameters:
    - smiles_query: SMILES string of the query molecule.
    - smiles_list: List of SMILES strings of the target molecules.
    - fingerprint_bits: Number of bits for Morgan fingerprints.
    - sanitize_mols: Whether to sanitize molecules.
    
    Returns:
    - tanimoto_similarities: List of Tanimoto similarity values (based on Morgan fingerprints).
    - murcko_similarities: List of Murcko scaffold-based Tanimoto similarity values.
    - gobbi_similarities: List of Gobbi similarity values (based on MACCS keys).
    """
    # Convert the query molecule to an RDKit molecule object
    query_mol = Chem.MolFromSmiles(smiles_query)
    if sanitize_mols:
        Chem.SanitizeMol(query_mol)
    
    # Tanimoto similarity: Calculate Morgan fingerprint for the query molecule
    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, radius=3, nBits=fingerprint_bits)
    
    # Murcko scaffold similarity: Calculate Murcko scaffold and its fingerprint
    query_scaffold = MurckoScaffold.GetScaffoldForMol(query_mol)
    query_scaffold_fp = AllChem.GetMorganFingerprintAsBitVect(query_scaffold, radius=3, nBits=fingerprint_bits)

    # Gobbi similarity: Calculate MACCS keys for the query molecule
    query_gobbi_fp = MACCSkeys.GenMACCSKeys(query_mol)
    
    tanimoto_similarities = []
    murcko_similarities = []
    gobbi_similarities = []

    for smiles in smiles_list:
        # Convert each target molecule to an RDKit molecule object
        target_mol = Chem.MolFromSmiles(smiles)
        if target_mol is None:
            continue
            
        if sanitize_mols:
            Chem.SanitizeMol(target_mol)
        
        # Tanimoto similarity
        target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, radius=3, nBits=fingerprint_bits)
        tanimoto_similarity = TanimotoSimilarity(query_fp, target_fp)
        tanimoto_similarities.append(tanimoto_similarity)

        # Murcko scaffold similarity
        target_scaffold = MurckoScaffold.GetScaffoldForMol(target_mol)
        target_scaffold_fp = AllChem.GetMorganFingerprintAsBitVect(target_scaffold, radius=3, nBits=fingerprint_bits)
        murcko_similarity = TanimotoSimilarity(query_scaffold_fp, target_scaffold_fp)
        murcko_similarities.append(murcko_similarity)
        
        # Gobbi similarity (MACCS keys)
        target_gobbi_fp = MACCSkeys.GenMACCSKeys(target_mol)
        gobbi_similarity = TanimotoSimilarity(query_gobbi_fp, target_gobbi_fp)
        gobbi_similarities.append(gobbi_similarity)

    return tanimoto_similarities, murcko_similarities, gobbi_similarities


def load_smiles_data(mode, folder_path, csv_file=None):
    """Load SMILES data based on the analysis mode."""
    if mode == "docking":
        # Load from pickle file
        filepath = os.path.join(folder_path, "my_dict.pkl")
        with open(filepath, "rb") as file:
            data = pickle.load(file)
        return [smile[-1] for smile in data]
    
    elif mode == "docking_txt":
        # Load from text file
        file_path = "smiles_list.txt"
        with open(file_path, "r") as file:
            data = file.readlines()
        return [smile.strip() for smile in data]
    
    elif mode in ["chem", "squid"]:
        # Load from CSV file
        if csv_file is None:
            csv_file = 'example.csv' if mode == "chem" else 'molecule_data.csv'
        df = pd.read_csv(csv_file)
        groups = df.groupby('target')
        target_smiles = {target: group['smiles'].tolist() for target, group in groups}
        return target_smiles
    
    elif mode == "hit_expansion":
        # Load from text file in hit_expansion directory
        filepath = os.path.join(folder_path, "hit_expansion", "moleculess.txt")
        with open(filepath, "r") as file:
            data = file.readlines()
        return [smile.strip() for smile in data]
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def get_docked_file_pattern(mode, folder_path, index):
    """Get the docked file pattern based on mode."""
    patterns = {
        "docking": f"r{index}_docked.sdf",
        "docking_txt": f"r{index}_docked.sdf", 
        "chem": f"c{index}_docked.sdf",
        "squid": f"sq_{index}_docked.sdf",
        "hit_expansion": f"{index}_docked.sdf"
    }
    
    if mode == "hit_expansion":
        return os.path.join(folder_path, "hit_expansion", patterns[mode])
    else:
        return os.path.join(folder_path, patterns[mode])


def analyze_docking_results(mode, max_folders=17, csv_file=None, verbose=True):
    """
    Main analysis function that handles all different modes.
    
    Parameters:
    - mode: Analysis mode ('docking', 'docking_txt', 'chem', 'squid', 'hit_expansion')
    - max_folders: Maximum number of folders to process
    - csv_file: CSV file for chem/squid modes
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
    
    # Initialize result containers
    results = []
    similarities_results = []
    docking_results = []
    temporal_mols = []
    
    for i, folder in enumerate(folders[:max_folders]):
        if verbose:
            print(f"Processing PDBID: {folder[-4:]}")
        
        # Check if initial docked file exists
        out_path = os.path.join(folder, '0_docked.sdf')
        if not os.path.exists(out_path):
            if verbose:
                print("Empty - skipping")
            continue
        
        # Load ligand
        try:
            lig = Chem.rdmolfiles.MolFromMol2File(ligands[i], sanitize=True)
            if lig is None:
                continue
            temporal_mols.append(lig)
            centroid = ComputeCentroid(lig.GetConformer())
            centroid = [centroid.x, centroid.y, centroid.z]
        except Exception as e:
            if verbose:
                print(f"Error loading ligand: {e}")
            continue
        
        # Set up file paths
        crys = os.path.join(folder, "dock_pose_crystal.sdf")
        crysmol2 = os.path.join(folder, f"{folder[-4:]}_ligand.mol2")
        
        # Load SMILES data based on mode
        try:
            if mode in ["chem", "squid"]:
                target_smiles_dict = load_smiles_data(mode, folder, csv_file)
                target = list(target_smiles_dict.keys())[i]
                smiles_list = target_smiles_dict[target]
            else:
                smiles_list = load_smiles_data(mode, folder)
        except Exception as e:
            if verbose:
                print(f"Error loading SMILES data: {e}")
            continue
        
        # Process docking results
        ds = {}
        all_scores = []
        
        for idx, smile in enumerate(smiles_list):
            docked_file = get_docked_file_pattern(mode, folder, idx)
            
            if not os.path.exists(docked_file):
                continue
            
            try:
                with open(docked_file, "r") as file:
                    lines = file.readlines()
                
                if len(lines) == 0:
                    continue
                
                scores = []
                for line_idx, line in enumerate(lines):
                    split_line = line.split()
                    if (split_line and split_line[-1] == "<minimizedAffinity>" 
                        and line_idx + 1 < len(lines) and float(lines[line_idx + 1]) < 0):
                        scores.append(float(lines[line_idx + 1]))
                        all_scores.append(float(lines[line_idx + 1]))
                
                # Pad to 10 scores with NaN if needed
                while len(scores) < 10:
                    scores.append(np.nan)
                
                ds[smile] = scores
                
            except Exception as e:
                if verbose:
                    print(f"Error processing docked file {docked_file}: {e}")
                continue
        
        # Get reference crystal docking score
        try:
            with open(crys, "r") as file:
                lines = file.readlines()
            
            crystal_mol = Chem.MolFromMol2File(crysmol2)
            if crystal_mol is None:
                continue
                
            if mode != "docking_txt":
                Chem.SanitizeMol(crystal_mol)
            
            ref_smiles = Chem.MolToSmiles(crystal_mol, canonical=True)
            
            crystal_score = None
            for line_idx, line in enumerate(lines):
                split_line = line.split()
                if split_line and split_line[-1] == "<minimizedAffinity>":
                    crystal_score = float(lines[line_idx + 1])
                    break
                    
        except Exception as e:
            if verbose:
                print(f"Error processing crystal structure: {e}")
            continue
        
        # Calculate similarities
        try:
            fingerprint_bits = 4096 if mode == "hit_expansion" else 1024
            sanitize = mode == "chem"
            
            if mode in ["chem", "squid"]:
                sim, msim, gsim = calculate_all_similarities(target, smiles_list, 
                                                           fingerprint_bits, sanitize)
            else:
                sim, msim, gsim = calculate_all_similarities(ref_smiles, smiles_list, 
                                                           fingerprint_bits, sanitize)
            
            # Calculate molecular properties
            if mode in ["chem", "squid"]:
                prop = calculate_molecule_properties(target, smiles_list)
            else:
                prop = calculate_molecule_properties(ref_smiles, smiles_list)
                
        except Exception as e:
            if verbose:
                print(f"Error calculating similarities: {e}")
            continue
        
        # Analyze results
        if ds:
            ds_df = pd.DataFrame(ds)
            min_scores = ds_df.min(axis=0)
            mean_score = np.mean(min_scores)
            
            if verbose:
                if crystal_score is not None:
                    print(f"Reference crystal dock: {round(crystal_score, 2)}")
                print(f"Generated dock: {round(mean_score, 2)} ± {round(np.var(min_scores), 2)}, min: {round(np.min(min_scores), 2)}")
                print(f"Tanimoto similarities: {round(np.mean(sim), 2)} ± {round(np.std(sim), 2)}")
                print(f"Murcko similarities: {round(np.mean(msim), 2)} ± {round(np.std(msim), 2)}")
                print(f"Gobbi similarities: {round(np.mean(gsim), 2)} ± {round(np.std(gsim), 2)}")
            
            # Store results
            result = {
                'pdb_id': folder[-4:],
                'crystal_score': crystal_score,
                'mean_generated_score': mean_score,
                'min_generated_score': np.min(min_scores),
                'score_variance': np.var(min_scores),
                'tanimoto_mean': np.mean(sim),
                'tanimoto_std': np.std(sim),
                'murcko_mean': np.mean(msim),
                'murcko_std': np.std(msim),
                'gobbi_mean': np.mean(gsim),
                'gobbi_std': np.std(gsim),
                'num_molecules': len(smiles_list)
            }
            
            if mode == "hit_expansion" and crystal_score is not None:
                # Calculate proportion better than crystal
                count_better = np.sum(min_scores < crystal_score)
                proportion_better = count_better / len(min_scores)
                result['proportion_better_than_crystal'] = proportion_better
                if verbose:
                    print(f"Proportion better than crystal: {proportion_better:.3f}")
            
            results.append(result)
            similarities_results.append(sim)
            if crystal_score is not None:
                docking_results.append(mean_score - crystal_score)
    
    # Print summary statistics
    if results and verbose:
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        
        if mode in ["chem", "squid"]:
            max_sims = [np.max(sim) for sim in similarities_results if len(sim) > 0]
            if max_sims:
                print(f"Mean max Tanimoto similarity: {np.mean(max_sims):.3f}")
        
        valid_docking = [x for x in docking_results if not np.isnan(x)]
        if valid_docking:
            print(f"Mean docking improvement: {np.mean(valid_docking):.3f}")
        
        if mode == "hit_expansion":
            hit_proportions = [r['proportion_better_than_crystal'] for r in results 
                             if 'proportion_better_than_crystal' in r]
            if hit_proportions:
                print(f"Mean proportion better than crystal: {np.mean(hit_proportions):.3f}")
    
    # Save results
    if temporal_mols:
        with open("ligands.pkl", "wb") as file:
            pickle.dump(temporal_mols, file)
        if verbose:
            print(f"Saved {len(temporal_mols)} ligand structures to ligands.pkl")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Unified Analysis Script for Synthformer Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--mode', 
        required=True,
        choices=['docking', 'docking_txt', 'chem', 'squid', 'hit_expansion'],
        help='Analysis mode to run'
    )
    
    parser.add_argument(
        '--max-folders',
        type=int,
        default=17,
        help='Maximum number of folders to process (default: 17)'
    )
    
    parser.add_argument(
        '--csv-file',
        type=str,
        help='CSV file for chem/squid modes (default: example.csv for chem, molecule_data.csv for squid)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file to save results (JSON format)'
    )
    
    args = parser.parse_args()
    
    # Run analysis
    results = analyze_docking_results(
        mode=args.mode,
        max_folders=args.max_folders,
        csv_file=args.csv_file,
        verbose=not args.quiet
    )
    
    # Save results if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main() 