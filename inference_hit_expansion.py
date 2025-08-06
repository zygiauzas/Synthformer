import torch 
from Dataloader import Datasetp4,custom_collate_fn
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from model import Transformer
import tqdm
import torch
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import pandas as pd
import pickle
from pathlib import Path
import sys, os

import numpy as np
from tqdm import tqdm
import pandas as pd
from rdkit.Chem import rdChemReactions
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
import random
import os
import os
import os

# Load datap4 from a pickle file
with open('datap4.pkl', 'rb') as f:
    datap4 = pickle.load(f)

# Load data from a pickle file
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)
from utils import load_enamine_building_blocks
filtered_cl = load_enamine_building_blocks(max_length=30)
compound_list = filtered_cl.copy()  # Keep compound_list for backward compatibility

bb= [Chem.MolFromSmiles(m) for m in filtered_cl]

# bb=[Chem.AddHs(m) for m in bb]
bbvocablen=len(bb)

def process_reactions_autoregressive_hit_expansion(p4, start_token_mf, rxn, model, bb, fingerprint_list, hit_molecule, compound_list=None, max_steps=10, end_token_id=None):
    """
    Autoregressive reaction processing for hit expansion:
    Uses a known hit molecule as one of the reactants in synthesis
    
    Args:
    - p4: Input pharmacophore tensor
    - start_token_mf: Start token molecular fingerprint
    - rxn: List of reaction objects
    - model: Trained transformer model
    - bb: List of building block molecules
    - fingerprint_list: List of building block fingerprints
    - hit_molecule: Known hit molecule to use in reactions
    - compound_list: List of building block SMILES
    - max_steps: Maximum number of synthesis steps
    - end_token_id: ID for end token
    
    Returns:
    - final_molecules: List of synthesized molecules
    - synthesis_paths: List of synthesis pathways
    - success: Boolean indicating if synthesis completed successfully
    """
    
    molecules = []
    synthesis_paths = []
    
    print("Starting autoregressive hit expansion...")
    
    for attempt in range(max_steps):
        print(f"Hit expansion attempt {attempt + 1}/{max_steps}")
        
        # Initialize for this attempt
        mflist = start_token_mf.clone()
        current_path = []
        
        try:
            # Predict building block and reaction
            bb_pred, bb_fingerprint, reaction_pred = model.predict(
                p4, mflist, fingerprint_list, end_token_id=end_token_id
            )
            
            # Check for end token
            if bb_pred is None or (end_token_id is not None and bb_pred[0] == end_token_id):
                print("End token generated.")
                continue
                
            # Try multiple building blocks from prediction
            for bb_idx in bb_pred:
                if bb_idx >= len(bb):
                    continue
                    
                building_block = bb[bb_idx]
                building_block_smiles = compound_list[bb_idx] if compound_list else Chem.MolToSmiles(building_block)
                
                print(f"Trying building block: {building_block_smiles}")
                
                # Try reactions with the hit molecule
                for reaction_idx in reaction_pred:
                    if reaction_idx >= len(rxn):
                        continue
                        
                    reaction = rxn[reaction_idx]
                    reaction.Initialize()
                    
                    try:
                        # Check if building block and hit molecule can react
                        if (reaction.IsMoleculeReactant(building_block) and 
                            reaction.IsMoleculeReactant(hit_molecule)):
                            
                            print(f"Applying reaction {reaction_idx}")
                            
                            # Run the reaction
                            products_tuple = reaction.RunReactants((hit_molecule, building_block))
                            
                            if products_tuple and len(products_tuple) > 0 and len(products_tuple[0]) > 0:
                                product = products_tuple[0][0]
                                Chem.SanitizeMol(product)
                                product_smiles = Chem.MolToSmiles(product)
                                
                                print(f"Reaction successful: {product_smiles}")
                                
                                molecules.append(product_smiles)
                                current_path.append((building_block_smiles, reaction_idx, product_smiles))
                                synthesis_paths.append(current_path.copy())
                                
                                # Break after successful reaction for this attempt
                                break
                                
                    except Exception as e:
                        print(f"Reaction failed: {e}")
                        continue
                        
                # If we found a successful reaction, move to next attempt
                if current_path:
                    break
                    
        except Exception as e:
            print(f"Prediction failed: {e}")
            continue
    
    success = len(molecules) > 0
    return molecules, synthesis_paths, success

def process_reactions(p4, mflist, rxn, model,bb,a,states=False,compound_list=None):
    """
    Process reactions iteratively using a machine learning model.
    
    Args:
    - p4: Input tensor for the model.
    - mflist: Tensor containing the molecular fingerprints.
    - rxn: List of reaction objects.
    - model: Machine learning model for predicting building blocks and reactions.
    - bb: List of building blocks.
    
    Returns:
    - Updated molecular fingerprint list.
    """
    c=0
    d=0
    reactant=[]
    molecules=[]
    reaction=[]
    reactions=[]
    for i in range(50):
        state=False

        for bbbb in range(5):
            if state==True:
                print("end",i)
                print("reactant",reactant)
                break

            # print(p4.shape, mflist.shape)
            bbout,buildingblockmf, reaction_pred = model.predict(p4, mflist, fingerprint_list, end_token_id=bbvocablen+1)
            # pred1=compound_list[bbout+1]
            # print(i)
            for bbb in bbout:
                if bbb-1>len(bb):
                    continue
                try:
                    pred2=bb[bbb]
                except:
                    print("failed")
                    continue
                # molecules.append(compound_list[bbout[0]]) 
                mol1=mflist[-1]
                mol2=pred2
                d+=1
                for ii,ir in enumerate(reaction_pred):
                    rx=rxn[ir]
                    rx.Initialize()
                    reactions.append(ii)

                    try:
                        if rx.IsMoleculeReactant(mol2) and rx.IsMoleculeReactant(a):
                            print("wow2",ii, ir)
                            if ii<5:
                                c+=1
                            prod=rx.RunReactants((a, mol2))
                            print(prod)
                            if len(prod)==0:
                                continue
                            print("there is a prod",prod)
                            prod=prod[0][0]
                            Chem.SanitizeMol(prod)
                            reactant.append(compound_list[bbout[0]])
                            reaction.append(ii)
                            # prod.SanitizeMol()
                            sm=Chem.MolToSmiles(prod)
                            prod=Chem.MolFromSmiles(sm)
                            molecules.append(Chem.MolToSmiles(prod)) 
                            morgan_fp = AllChem.GetMorganFingerprintAsBitVect(prod, useChirality=True, radius=5, nBits=1024)
                            morgan_tensor = torch.Tensor(np.array(morgan_fp))
                            if morgan_tensor.dim() == 1:
                                morgan_tensor = morgan_tensor.unsqueeze(0)
                                morgan_tensor.unsqueeze_(0)
                            print(mflist.shape, morgan_tensor.shape)
                            print(molecules)
                            print("reactant", reactant)
                            print("molecules", molecules)
                            print(reaction,ii)
                            mflist = torch.cat((mflist, morgan_tensor), dim=-2)
                            states=True
                            state=True
                            
                    except Exception as e:
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)
                        print(e)
                        print("failed reaction")
            
            if state==True:
                break
                
        print("final molecule")
    return mflist,molecules,reactions,states

# Initialize the transformer model
model = Transformer(
        source_vocab_size=100,
        target_vocab_size=bbvocablen+2+1,
        embedding_dim=512,
        source_max_seq_len=256,
        target_max_seq_len=256,
        num_layers=7,
        num_heads=8,
        dropout=0.1
    )

# Load the model from the pickle file
model_path = 'modele.pth'
model.load_state_dict(torch.load(model_path))
# Set the model to evaluation mode
model.eval()
fingerprint_list = []
for item in bb:
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(item, 2, nBits=1024)
    fingerprint_array = np.array(list(fingerprint.ToBitString())).astype(np.float32)
    fingerprint_tensor = torch.from_numpy(fingerprint_array)
    fingerprint_list.append(fingerprint_array)
bbvocablen = len(bb)
list_index = list(range(len(datap4)))
train_data, test_data, _, _ = train_test_split(list_index, list_index, test_size=0.2, random_state=42)
train_data, val_data, _, _ = train_test_split(train_data, train_data, test_size=0.15, random_state=42)
print(len(train_data), len(val_data), len(test_data))

mflist = torch.tensor([[[1] + [0] * 1023]]).float()
df = pd.read_excel('reactions.xls')
reactions=df["smirks"].to_list()
rxn=[rdChemReactions.ReactionFromSmarts(r) for r in reactions]
bimolar = [r for r in rxn if r.GetNumReactantTemplates() == 2]
rxn=bimolar
state=False
m=[]
r=[]
folder_path = "refined-set/"
folders = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
print(folders[:10])
folders.sort()
for folder in folders[:17]:
    m=[]
    # Open each folder
    print(folder)
    folder=os.path.join(folder)

    folder_contents = os.listdir(folder+'/')
    # mol2=os.path.join(folder+'/'+folder_contents[0])

    print(folder)
    folder_contents = os.listdir(folder+'/')
    # mol2=os.path.join(folder+'/'+folder_contents[0])
    for sdf_file in folder_contents:
        if sdf_file.endswith("ligand.mol2"):
            sdf=sdf_file
            break

    folder_f=folder+'/'+sdf
    print(folder_f)
    try:
        # a=Chem.MolFromMolFile(folder_f,sanitize=False)
        a=Chem.MolFromMol2File(folder_f)
        Chem.SanitizeMol(a)
        # a=Chem.RemoveHs(a)
        smilei=Chem.MolToSmiles(a, canonical=True)
        print(smilei)
        # Calculate the Morgan fingerprint for molecule a
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(a, 2, nBits=1024)
        fingerprint_array = np.array(list(fingerprint.ToBitString())).astype(np.float32)
        fingerprint_tensor = torch.from_numpy(fingerprint_array)


    except Exception as e:
    # Handle the exception
        print(f"An error occurred: {e}")
        print("failed")
        continue
    
    # Convert datap4 to a torch tensor
    try:
        p4 = torch.load(folder + "/datap4_tensor.pt").to(torch.float)
    except:
        print("fail")
        continue
    moleculess=[]
    for b in range(100):
        if len( moleculess)>100:
            break
        print("start inference", len(mol))
        mflist = torch.tensor([[[1] + [0] * 1023]]).float()
        mflist = torch.cat((mflist, fingerprint_tensor.unsqueeze(0).unsqueeze(0)), dim=1)

        
        mflist,molecules,reactions,state = process_reactions(p4, mflist, rxn, model,bb,a,False,compound_list)
        
        # Alternative: Use new autoregressive approach (commented out for backward compatibility)
        # start_token = torch.tensor([[[1] + [0] * 1023]]).float()
        # final_molecules, synthesis_paths, success = process_reactions_autoregressive_hit_expansion(
        #     p4=p4, 
        #     start_token_mf=start_token,
        #     rxn=rxn, 
        #     model=model, 
        #     bb=bb, 
        #     fingerprint_list=fingerprint_list,
        #     hit_molecule=a,
        #     compound_list=compound_list,
        #     max_steps=50,
        #     end_token_id=bbvocablen+1
        # )
        # if success:
        #     molecules = final_molecules
        if len(molecules)==0:
            continue
        moleculess.extend(molecules)
        # except Exception as e:
        #     exc_type, exc_obj, exc_tb = sys.exc_info()
        #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        #     print(exc_type, fname, exc_tb.tb_lineno)
        #     print("failed reaction")
        #     continue
    # Specify the new folder path
    if len(moleculess)==0:
        continue
    
    new_folder = folder+"/hit_expansion/"
    print(new_folder)

    # Check if new_folder exists
    if not os.path.exists(new_folder):
        # Create a new folder
        os.makedirs(new_folder)


    # Create the file path
    file_path = new_folder + "moleculess.txt"

    # Open the file in write mode
    with open(file_path, 'w') as f:
        # Write each string in moleculess to the file
        for molecule in moleculess:
            f.write(molecule + "\n")


# Specify the folder path

# Create the file path
file_path = folder + "/output.pkl"

# Open the file in write mode
with open(folder+'/my_dict.pkl', 'wb') as f:
    pickle.dump(m, f)

    # Create the file path
    file_path = new_folder + "moleculess.txt"

    # Open the file in write mode
    with open(file_path, 'w') as f:
        # Write each string in moleculess to the file
        for molecule in moleculess:
            f.write(molecule + "\n")

    print(new_folder)
    print(len(m))
# Specify the folder path


    # Create the file path
    file_path = folder+ "/output.pkl"



    # Open the file in write mode
    with open(folder+'/my_dict.pkl', 'wb') as f:
        pickle.dump(m, f)

    
print(folder)




    