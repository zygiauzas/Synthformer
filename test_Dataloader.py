import pytest
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
import torch.nn.functional as F
from rdkit.Chem import rdChemReactions
import pandas as pd

def check_dataloader():
    # Load datap4 from a pickle file
    with open('datap4.pkl', 'rb') as f:
        datap4 = pickle.load(f)

    # Load data from a pickle file
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)

    list_index=list(range(len(datap4)))

    with open("mcule_bb.smi") as f:
        line = f.readlines(1000000)
        compound_list=[a.split('\t')[0] for a in line]
        filtered_cl=[]
        for mol in compound_list:
            if len(mol)<30:
                filtered_cl.append(mol)

    filtered_cl=[Chem.CanonSmiles(m) for m in filtered_cl]
    compound_list=filtered_cl
    

    bb= [Chem.MolFromSmiles(m,sanitize=True) for m in compound_list]
    bbvocablen=len(bb)

    train_data, test_data, _, _ = train_test_split(list_index, list_index, test_size=0.3, random_state=42)
    train_data, val_data, _, _ = train_test_split(train_data, train_data, test_size=0.15, random_state=108)
    print(len(train_data), len(val_data), len(test_data))
    fingerprint_list = []
    for item in bb:
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(item, 2, nBits=1024)
        fingerprint_array = np.array(list(fingerprint.ToBitString())).astype(np.float32)
        fingerprint_tensor = torch.from_numpy(fingerprint_array)
        fingerprint_list.append(fingerprint_array)

    # fingerprint_tensor = torch.stack(fingerprint_list)
    dataset=Datasetp4(data,datap4,train_data,fingerprint_list)
    train_load=DataLoader(dataset,collate_fn=custom_collate_fn,batch_size=1)
    df = pd.read_excel('reactions.xls')
    reactions=df["smirks"].to_list()
    rxn=[rdChemReactions.ReactionFromSmarts(r) for r in reactions]
    rxn = [r for r in rxn if r.GetNumReactantTemplates() == 2]
    for batch in train_load:
        reactions = batch[1]
        mflist = batch[2]
        buildingblock = batch[3]
        buildingblockmf = batch[4]
        smiles=batch[6]
        if reactions.size(1)==4:
            print("smiles",smiles)
            print(buildingblock)
            print(reactions)
            print(reactions.tolist())
            buildingblock=buildingblock.tolist()
            rec=reactions.tolist()[0]
            for i,a in enumerate(rec):
                print("wow",i)
                print([filtered_cl[int(b)] for b in buildingblock[0][:-1]])
                # print(buildingblock)
                print(buildingblock[0][i])
                print(buildingblock[0][i+1])
                print("bb1: ",compound_list[int(buildingblock[0][i])] )
                
                print(int(rec[i+1]))
                if i==0:
                    #  
                    # print("bb2: ",compound_list[int(buildingblock[0][i+1])] )
                     mol1=bb[int(buildingblock[0][i])]
                else:
                    print('loaded correctly')
                    mol1=prod[0][0]
                    Chem.SanitizeMol(mol1)
                    Chem.Kekulize(mol1)
                print(buildingblock[0][i+1],len(bb))
                if int(buildingblock[0][i+1])>len(bb):
                    print("product: ",Chem.CanonSmiles(smiles[0]))
                    print(smileprod)
                    return smileprod,Chem.CanonSmiles(smiles[0])
                    
                # mol1=bb[int(buildingblock[0][i])]
                mol2=bb[int(buildingblock[0][i+1])]
                prod=rxn[int(rec[i+1])].RunReactants((mol1,mol2))
              
                print(prod)
                print(prod[0])
                smileprod=Chem.CanonSmiles(Chem.MolToSmiles(prod[0][0]))
                print(smileprod)
            


def test_capital_case():
    p1,p2=check_dataloader()
    assert p1== p2
    

