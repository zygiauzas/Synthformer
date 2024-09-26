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

# Load datap4 from a pickle file
with open('datap4.pkl', 'rb') as f:
    datap4 = pickle.load(f)

# Load data from a pickle file
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)
filtered_cl=[]
with open("mcule_bb.smi") as f:
    line = f.readlines(1000000)
    compound_list=[a.split('\t')[0] for a in line]
    for mol in compound_list:
        if len(mol)<30:
            filtered_cl.append(mol)

bb= [Chem.MolFromSmiles(m) for m in filtered_cl]

# bb=[Chem.AddHs(m) for m in bb]
bbvocablen=len(bb)
def process_reactions(p4, mflist, rxn, model,bb,states=False,compound_list=None):
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
                            
                    except Exception as e:
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)
                        print(e)
                        print("failed reaction")
                        
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

    # Convert datap4 to a torch tensor
    try:
        p4 = torch.load(folder + "/datap4_tensor.pt").to(torch.float)
    except:
        print("fail")
        continue

    while len(m)<100:
        print("start inference")
        mflist = torch.tensor([[[1] + [0] * 1023]]).float()
        # try
        mflist,molecules,reactions,state = process_reactions(p4, mflist, rxn, model,bb,False,compound_list)
        if len(molecules)==0:
            continue
        m.append(molecules)
        # except Exception as e:
        #     exc_type, exc_obj, exc_tb = sys.exc_info()
        #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        #     print(exc_type, fname, exc_tb.tb_lineno)
        #     print("failed reaction")
        #     continue

    print(len(m))
# Specify the folder path


    # Create the file path
    file_path = folder+ "/output.pkl"



    # Open the file in write mode
    with open(folder+'/my_dict.pkl', 'wb') as f:
        pickle.dump(m, f)

    
print(folder)




    