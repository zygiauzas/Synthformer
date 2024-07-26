from os import listdir
from os.path import isfile, join
import os
import sys

import warnings
from pathlib import Path
import subprocess

# import nglview as nv
from openbabel import pybel
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from rdkit.Chem.rdMolTransforms import ComputeCentroid
from rdkit import Chem

from opencadd.structure.core import Structure
import pickle

sys.path.insert(0, './../')


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
    mol = pybel.readfile("sdf", smi_path)
    for molecule in mol:
        molecule=molecule
        break
#     molecule = pybel.readstring("sdf", smi_path)
    # add hydrogens at given pH
    molecule.OBMol.CorrectForPH(pH)
    molecule.addh()
    # generate 3D coordinates
#     molecule.make3D(forcefield="mmff94s", steps=10000)
    # add partial charges to each atom
    for atom in molecule.atoms:
        atom.OBAtom.GetPartialCharge()
    molecule.write("pdbqt", str(pdbqt_path), overwrite=True)
    return

def run_smina(
    ligand_path, protein_path, out_path, pocket_center, pocket_size,atom_terms, num_poses=10, exhaustiveness=8
):
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
            "--ligand",
            str(ligand_path),
            "--receptor",
            str(protein_path),
            "--out",
            str(out_path),
            "--center_x",
            str(pocket_center[0]),
            "--center_y",
            str(pocket_center[1]),
            "--center_z",
            str(pocket_center[2]),
            "--size_x",
            str(pocket_size[0]),
            "--size_y",
            str(pocket_size[1]),
            "--size_z",
            str(pocket_size[2]),
            "--num_modes",
            str(num_poses),
            "--exhaustiveness",
            str(exhaustiveness),
            "--atom_terms",
            atom_terms
        ],
        universal_newlines=True,  # needed to capture output text
    )
    return output_text

def minimise_smina(
    ligand_path, protein_path, out_path, atom_terms
):
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
            "--ligand",
            str(ligand_path),
            "--receptor",
            str(protein_path),
            "--out",
            str(out_path),
            "--score_only",
            "--atom_terms",
            str(atom_terms)
        ],
        universal_newlines=True,  # needed to capture output text
    )
    return output_text
thisdir = os.getcwd()

mypath=thisdir+"/refined-set/"
folder=mypath
folder_path = thisdir+"/refined-set/"
folders = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

folders.sort()
ligands=[folder+"/"+folder[-4:]+"_ligand.mol2" for folder in folders]
proteins=[folder+"/"+folder[-4:]+"_protein.pdb" for folder in folders]
print("proteins",proteins[:10])
for i,a in enumerate([folders[0]]):
    print(i, 'start preparation', a)
    lig=Chem.rdmolfiles.MolFromMol2File(ligands[i], sanitize=True)
    centroid = ComputeCentroid(lig.GetConformer())
    centroid=[centroid.x, centroid.y, centroid.z]
    mol=a+'/mol.pdbqt'
    prot=proteins[i]+"qt"
    # print(ligands[i][:-4]+'sdf')
    try:
        
        sdf_to_pdbqt(ligands[i][:-4]+'sdf',mol)
        gen_mol_dir=a
    
        box=[25,25,25]
        print("start minimising")
        print(proteins[i])
        print(prot[i])
        pdb_to_pdbqt(proteins[i],prot)
        print("pdb converted")
        minimise_smina(mol, prot, a+"/dock_pose_crystal.sdf",a+"/mol_int.pdb")
        print("start protein to pdbqt")
        

        # Specify the filepath of the data.pkl file
        filepath = a+"/my_dict.pkl"
        print(filepath)
        # Open the file in read mode
        with open(filepath, "rb") as file:
            # Load the data from the pickle file
            data = pickle.load(file)

# Now you can use the 'data' variable to access the contents of the pickle file
        print(data)
        for b,smile in enumerate(data):
            smi_path=a+'/'+str(b)+'_mol.pdbqt'
            # print(smile, smi_path)
            print("start smile to pdbqt")
            print(smile[-1],smi_path)
            smiles_to_pdbqt(smile[-1],smi_path)
            out_path=gen_mol_dir+"/"+str(b)+'_docked.sdf'
            print("start dock")
            run_smina(smi_path,prot,out_path,centroid,box,gen_mol_dir+str(b)+'_docked_mol_inter.pdb')
            print("finish dock")
    except:
        print("Fail", a )
   
 
        
    
    