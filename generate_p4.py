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

file_path = "data.pkl"

if __name__ == "__main__":
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    smiles=data[1]
    print(smiles)
    a=0
    datap4=[]
    for i in tqdm(range(1,len(smiles))):
        moli=Chem.MolFromSmiles(smiles[i])
        Chem.Kekulize(moli)
        smilei=Chem.MolToSmiles(moli)
        o3d=generate_3d(smilei)
        if o3d is None:
            continue
        p4=get_pharmacophores(smilei,o3d)
        datap4.append(prep_data(o3d,p4))
        a+=1
    print(np.round(a/len(smiles)*100,2),"%")

# Save datap4 as a pickle file
    output_file_path = "datap4.pkl"
    with open(output_file_path, "wb") as file:
        pickle.dump(datap4, file)
        

        


# Now you can work with the loaded data
