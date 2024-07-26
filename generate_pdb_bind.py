
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import os
import os
from rdkit import Chem
import pickle
import os
import numpy as np
import pandas as pd
import rdkit 
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem import BRICS
from rdkit.Chem import rdMolDescriptors
from rdkit import RDConfig
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import BRICS
import pickle
from tqdm import tqdm
import torch

def get_pharmacophores(smiles,o3d):
    # Convert OBMol to RDKit Mol
    
    mol = Chem.MolFromSmiles(smiles)

    # Create a ChemicalFeatures factory
    fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

    # Get the features for the molecule
    features = factory.GetFeaturesForMol(mol)
    

    # Create a dictionary to store atom-specific pharmacophore information
    atom_pharmacophores = {i: [] for i, _ in enumerate(o3d.GetAtoms())}

    # Iterate over features and associate with atoms
    for feat in features:
        atom_ids = feat.GetAtomIds()
        for atom_id in atom_ids:
            atom_pharmacophores[atom_id].append(
                feat.GetFamily()
            )

    return atom_pharmacophores

def ghose_filter(mol):
    
    if mol is None:
        return False  # Invalid SMILES string
    
    # Calculate molecular weight, logP, number of rotatable bonds, and PSA
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    psa = Descriptors.TPSA(mol)
    
    # Check if the molecule passes Ghose's filter criteria
    if 160 <= mw <= 480 and 0.4 <= logp <= 5.6 and num_rotatable_bonds <= 10 and psa <= 140:
        return True
    else:
        return False
    
def generate_3d(smiles):
    """
    Generate a 3D structure for a molecule using RDKit.

    Parameters:
    - smiles (str): SMILES representation of the molecule.

    Returns:
    - rdkit.Chem.Mol: RDKit molecule with 3D coordinates.
    """
    mol = Chem.MolFromSmiles(smiles)
    mol= Chem.AddHs(mol)
    Chem.SanitizeMol(mol)

    if mol is not None:
        # Generate 3D coordinates for the molecule
        try:
            AllChem.EmbedMolecule(mol, randomSeed=42)  # You can change the random seed
            # Optimize the 3D structure
            AllChem.MMFFOptimizeMolecule(mol)
            AllChem.ComputeGasteigerCharges(mol)
        except:
            print("failed conf")
            return None
        return mol
    else:
        print("Invalid SMILES representation.")
        return None
def one_hot_encode(item_list):
    """
    One-hot encodes a list of items based on predefined categories.

    Parameters:
    item_list (list): List of items to be one-hot encoded.

    Returns:
    numpy.ndarray: One-hot encoded array.

    Example:
    >>> item_list = [['Acceptor', 'Donor'], ['Donor', 'Aromatic']]
    >>> one_hot_encode(item_list)
    array([[1, 1, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 1]])
    """
    # Define the categories
    categories = {'Acceptor', 'Donor', 'LumpedHydrophobe', 'ZnBinder', 'PosIonizable', 'Hydrophobe', 'NegIonizable', 'Aromatic'}
    
    # Initialize the encoded array with zeros
    encoded_array = np.zeros(len(categories), dtype=int)
   
    

    # Encode each item in the item_list
    for sublist in item_list:
        # Encode each item in the sublist
        encoded_array += [1 if item in sublist else 0 for item in categories]

    return encoded_array
def prep_data(o3d,o3d_pharmacophores):
    pos=[]
    for i, atom in enumerate(o3d.GetAtoms()):
            positions = o3d.GetConformer().GetAtomPosition(i)
            
            # charge=0
            if atom.GetSymbol()!='H':
                one_hot = one_hot_encode(o3d_pharmacophores[i])
                coords = np.array([float(positions.x), float(positions.y), float(positions.z)])
                pos.append(np.concatenate([one_hot, coords]))
                # pos.append(np.cat([one_hot_encode( o3d_pharmacophores[i]), np.array([float(positions.x), float(positions.y), float(positions.z)])])
    return pos

folder_path = "refined-set/"
folders = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
print(folders[:10])
folders.sort()
datap4=[]
for folder in folders:
    # Open each folder
    folder=os.path.join(folder)
    print(folder)
    folder_contents = os.listdir(folder+'/')
    # mol2=os.path.join(folder+'/'+folder_contents[0])
    print(folder+'/'+folder_contents[0])
    a=Chem.MolFromMol2File(folder+'/'+folder_contents[0])
    smilei=Chem.MolToSmiles(a)
    p4=get_pharmacophores(smilei,a)
    datap4.append(prep_data(a,p4))
    print(smilei)
    print(a)
    print(datap4)
    print(folder)
    

    # Specify the folder path to save the tensor
    save_folder = "tensor_folder/"

    # Convert datap4 to a torch tensor
    tensor_datap4 = torch.tensor(datap4)
    print(folder + "/datap4_tensor.pt")
    # Save the tensor in the specified folder
    torch.save(tensor_datap4, folder + "/datap4_tensor.pt")
    exit()
