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

# Load datap4 from a pickle file
with open('datap4.pkl', 'rb') as f:
    datap4 = pickle.load(f)

# Load data from a pickle file
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)
from utils import load_enamine_building_blocks
filtered_cl = load_enamine_building_blocks(max_length=50)

bb= [Chem.MolFromSmiles(m) for m in filtered_cl]
# bb=[Chem.AddHs(m) for m in bb]
bbvocablen=len(bb)
def process_reactions_autoregressive(p4, start_token_mf, rxn, model, bb, fingerprint_list, compound_list=None, max_steps=10, end_token_id=None):
    """
    Autoregressive reaction processing following the inference protocol:
    1. Start with pharmacophore data and start token
    2. Predict building block B1, then reaction R0  
    3. Use fingerprint of B1 to predict B2, then reaction R1, apply reaction to get P1
    4. Use product P1 to predict B3, then reaction R2, apply reaction to get P2
    5. Continue until end token is generated
    
    Args:
    - p4: Input pharmacophore tensor
    - start_token_mf: Start token molecular fingerprint
    - rxn: List of reaction objects
    - model: Trained transformer model
    - bb: List of building block molecules
    - fingerprint_list: List of building block fingerprints
    - compound_list: List of building block SMILES
    - max_steps: Maximum number of synthesis steps
    - end_token_id: ID for end token
    
    Returns:
    - final_molecule: The final synthesized molecule
    - synthesis_path: List of (building_block, reaction, product) tuples
    - success: Boolean indicating if synthesis completed successfully
    """
    
    # Initialize
    mflist = start_token_mf.clone()  # Start with start token
    molecules = []
    reactions_used = []
    products = []
    synthesis_path = []
    current_molecule = None
    
    print("Starting autoregressive synthesis...")
    
    for step in range(max_steps):
        print(f"Step {step + 1}/{max_steps}")
        
        # Predict building block and reaction
        try:
            bb_pred, bb_fingerprint, reaction_pred = model.predict(
                p4, mflist, fingerprint_list, end_token_id=end_token_id
            )
            
            # Check for end token
            if bb_pred is None or (end_token_id is not None and bb_pred[0] == end_token_id):
                print("End token generated. Synthesis complete.")
                break
                
            bb_idx = bb_pred[0].item()
            
            # Check if building block index is valid
            if bb_idx >= len(bb):
                print(f"Invalid building block index: {bb_idx}")
                break
                
            # Get the building block molecule
            building_block = bb[bb_idx]
            building_block_smiles = compound_list[bb_idx] if compound_list else Chem.MolToSmiles(building_block)
            
            print(f"Predicted building block: {building_block_smiles}")
            molecules.append(building_block_smiles)
            
            # Update mflist with the building block fingerprint
            if bb_fingerprint is not None:
                mflist = torch.cat((mflist, bb_fingerprint), dim=-2)
            
            # For the first step, just store the building block
            if step == 0:
                current_molecule = building_block
                synthesis_path.append((building_block_smiles, None, building_block_smiles))
                continue
            
            # Apply reaction
            if reaction_pred is not None:
                reaction_idx = reaction_pred
                if reaction_idx < len(rxn):
                    reaction = rxn[reaction_idx]
                    reaction.Initialize()
                    
                    print(f"Applying reaction {reaction_idx}")
                    
                    try:
                        # Check if both molecules can react
                        if (reaction.IsMoleculeReactant(current_molecule) and 
                            reaction.IsMoleculeReactant(building_block)):
                            
                            # Run the reaction
                            products_tuple = reaction.RunReactants((current_molecule, building_block))
                            
                            if products_tuple and len(products_tuple) > 0 and len(products_tuple[0]) > 0:
                                product = products_tuple[0][0]  # Take first product
                                product_smiles = Chem.MolToSmiles(product)
                                
                                print(f"Reaction successful: {product_smiles}")
                                
                                # Update current molecule for next iteration
                                current_molecule = product
                                products.append(product_smiles)
                                reactions_used.append(reaction_idx)
                                
                                # Add to synthesis path
                                synthesis_path.append((building_block_smiles, reaction_idx, product_smiles))
                                
                                # Update mflist with product fingerprint for next prediction
                                # Generate fingerprint for the product
                                try:
                                    product_fp = AllChem.GetMorganFingerprintAsBitVect(product, 2, nBits=1024)
                                    product_fp_array = np.array(list(product_fp.ToBitString())).astype(np.float32)
                                    product_fp_tensor = torch.tensor(product_fp_array, dtype=torch.float32, device=mflist.device)
                                    product_fp_tensor = product_fp_tensor.unsqueeze(0).unsqueeze(0)
                                    mflist = torch.cat((mflist, product_fp_tensor), dim=-2)
                                except Exception as e:
                                    print(f"Failed to generate product fingerprint: {e}")
                                    break
                                    
                            else:
                                print("Reaction failed - no products generated")
                                synthesis_path.append((building_block_smiles, reaction_idx, None))
                                break
                        else:
                            print("Molecules cannot react with this reaction")
                            synthesis_path.append((building_block_smiles, reaction_idx, None))
                            break
                            
                    except Exception as e:
                        print(f"Reaction failed with error: {e}")
                        synthesis_path.append((building_block_smiles, reaction_idx, None))
                        break
                else:
                    print(f"Invalid reaction index: {reaction_idx}")
                    break
            else:
                print("No reaction predicted")
                break
                
        except Exception as e:
            print(f"Prediction failed at step {step}: {e}")
            break
    
    # Return results
    final_molecule = current_molecule
    final_smiles = Chem.MolToSmiles(final_molecule) if final_molecule else None
    success = final_molecule is not None and len(synthesis_path) > 0
    
    return final_smiles, synthesis_path, success

# Keep the original function for backward compatibility
def process_reactions(p4, mflist, rxn, model,bb,fingerprint_list,states=False,compound_list=None):
    """
    Original process_reactions function - kept for backward compatibility.
    Consider using process_reactions_autoregressive for the new inference protocol.
    """
    c=0
    d=0
    reactant=[]
    molecules=[]
    reaction=[]
    reactions=[]
    for i in range(7):
        if i>len(molecules)+1:
            return mflist,molecules,reactions,states
        state=False
        print("state",state)
        for bbbb in range(5):
            if state==True:
                print("end",i)
                print("reactant",reactant)
                break
            print("start",i)
            # print(p4.shape, mflist.shape)
            bbout,buildingblockmf, reaction_pred = model.predict(p4, mflist, fingerprint_list)
            print(bbout)
            mflist=torch.cat((mflist,buildingblockmf), dim=-2)
            print("bbout",bbout)

            if bbout[0]>len(bb)-1:
                print("end of the world")
                return mflist,molecules,reactions,states
            # pred1=compound_list[bbout+1]
            # print(i)
            if i==0:
                print(i)
                # molecules.append(compound_list[bbout[0]])   
                pred1=bb[bbout[0]]
                reactant.append(compound_list[bbout[0]])
                state=True
                continue
            
            if i>0:
                pred2=bb[bbout[0]]
                # molecules.append(compound_list[bbout[0]]) 
                if i==1:
                    mol1=pred1
                    mol2=pred2
                
                if i>1:
                    mol1=prod
                    mol2=pred2
                d+=1
                for ii,ir in enumerate(reaction_pred):
                    rx=rxn[ir]
                    rx.Initialize()
                    reactions.append(ii)
                    try:
                        if rx.IsMoleculeReactant(mol1) and rx.IsMoleculeReactant(mol2):
                            print("wow2",ii, ir)
                            if ii<5:
                                c+=1
                            prod=rx.RunReactants((mol1, mol2))
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
                            if len(molecules)==3:
                                return  mflist,molecules,reactions,states
                            break
                    except:
                        return  mflist,molecules,reactions,states
            
            if state==True:
                break
        # if len(molecules)<1:
        #     return None,None,None,None
                
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
fingerprint_tensor=torch.Tensor(fingerprint_list)
bbvocablen = len(bb)
list_index = list(range(len(datap4)))
train_data, test_data, _, _ = train_test_split(list_index, list_index, test_size=0.2, random_state=42)
train_data, val_data, _, _ = train_test_split(train_data, train_data, test_size=0.15, random_state=42)
print(len(train_data), len(val_data), len(test_data))
dataset = Datasetp4(data, datap4, test_data, fingerprint_list)
dataset=Datasetp4(data,datap4,test_data,fingerprint_list)
test_load=DataLoader(dataset,collate_fn=custom_collate_fn,batch_size=1)

for batch in test_load:
    
    p4 = batch[0]
    reactions = batch[1]
    mflist = batch[2]
    buildingblock = batch[3]
    buildingblockmf = batch[4]
    buildingblockmf = batch[5]
    mflist = torch.tensor([[[1] + [0] * 1023]]).float()
    df = pd.read_excel('reactions.xls')
    reactions=df["smirks"].to_list()
    rxn=[rdChemReactions.ReactionFromSmarts(r) for r in reactions]
    # bimolar = [r for r in rxn if r.GetNumReactantTemplates() == 2]
    # rxn=bimolar
    state=False
    m=[]
    r=[]
    for b in range(500):
        print("start inference")
        mflist = torch.tensor([[[1] + [0] * 1023]]).float()
        
        mflist,molecules,reactions,state = process_reactions(p4, mflist, rxn, model,bb,fingerprint_list,False,compound_list)
        
        
        if len(molecules)==0:
            continue
        print(molecules)
        
        
        m.append(molecules)
        # r.append(reactions)
        
    for i in m:
        print(i)
    print(len(m))
    exit()

def autoregressive_synthesis_example(model_path='modele.pth', p4_data=None):
    """
    Example usage of the new autoregressive synthesis protocol.
    
    Args:
    - model_path: Path to the trained model
    - p4_data: Pharmacophore data tensor
    
    Returns:
    - final_molecule: Final synthesized molecule SMILES
    - synthesis_path: Complete synthesis pathway
    """
    
    # Load model and data (this would typically be done elsewhere)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load building blocks and fingerprints
    from utils import load_enamine_building_blocks
    filtered_cl = load_enamine_building_blocks()
    bb = [Chem.MolFromSmiles(m) for m in filtered_cl]
    
    # Generate fingerprints
    fingerprint_list = []
    for item in bb:
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(item, 2, nBits=1024)
        fingerprint_array = np.array(list(fingerprint.ToBitString())).astype(np.float32)
        fingerprint_list.append(fingerprint_array)
    
    # Load model
    bbvocablen = len(fingerprint_list)
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
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load reaction templates (you would need to implement this)
    # rxn = load_reaction_templates()
    rxn = []  # Placeholder
    
    # Create start token (typically a special molecular fingerprint)
    start_token = np.zeros(1024)
    start_token[0] = 1  # Start token marker
    start_token_tensor = torch.tensor(start_token, dtype=torch.float32, device=device)
    start_token_mf = start_token_tensor.unsqueeze(0).unsqueeze(0)
    
    # Default p4 data if not provided
    if p4_data is None:
        p4_data = torch.randn(1, 256, 11, device=device)  # Example pharmacophore data
    
    # Run autoregressive synthesis
    end_token_id = bbvocablen + 1  # Define end token ID
    
    final_molecule, synthesis_path, success = process_reactions_autoregressive(
        p4=p4_data,
        start_token_mf=start_token_mf,
        rxn=rxn,
        model=model,
        bb=bb,
        fingerprint_list=fingerprint_list,
        compound_list=filtered_cl,
        max_steps=10,
        end_token_id=end_token_id
    )
    
    print(f"Synthesis {'successful' if success else 'failed'}")
    if final_molecule:
        print(f"Final molecule: {final_molecule}")
    
    print("Synthesis pathway:")
    for i, (bb_smiles, reaction_idx, product_smiles) in enumerate(synthesis_path):
        if reaction_idx is not None:
            print(f"Step {i+1}: {bb_smiles} + reaction {reaction_idx} -> {product_smiles}")
        else:
            print(f"Step {i+1}: Starting with {bb_smiles}")
    
    return final_molecule, synthesis_path

if __name__ == "__main__":
    # Example usage
    final_mol, path = autoregressive_synthesis_example()
    print(f"Generated molecule: {final_mol}")
    