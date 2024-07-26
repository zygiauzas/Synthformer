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
def Ghose_filter(mol):
    
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
def compute_prod(bb,rxns,run):
    data={}
    products_smile={}
    products_mf={}
    imf={}
    c=0
    products=[]
    # for ri, rxn in enumerate(rxns):
    #     rxn.Initialize()
    #     for b1,bb1 in enumerate(bb):
    #         if rxn.IsMoleculeReactant(bb1): 
    #             for b2,bb2 in enumerate(bb):
    #                 if  rxn.IsMoleculeReactant(bb2):
    #                     prod=rxn.RunReactants((bb2, bb1))
    #                     if prod==():
    #                         continue
    #                     prod=prod[0][0]
    #                     Chem.SanitizeMol(prod)
    #                     Chem.Kekulize(prod)
    #                     if not Ghose_filter(prod):
    #                         continue
    #                     products.append([prod,c])
    #                     data[c]=[ri,b1,b2,0]
    #                     products_smile[c]=Chem.MolToSmiles(prod)
    #                     products_mf[c]=np.array(AllChem.GetMorganFingerprintAsBitVect(prod,useChirality=True, radius=5, nBits = 1024))
    #                     c+=1
    for i,b in enumerate(bb):
        imf[i]=np.array(AllChem.GetMorganFingerprintAsBitVect(b,useChirality=True, radius=5, nBits = 1024))

    bb1=[[b,i] for i,b in enumerate(bb)]
    for l in range(6):
        print("done", len(products))
        prod2=[]
        for ri,rxn in enumerate(rxns):
            rxn.Initialize()
            count=0
            if l ==0:
                products=bb1
            for p in products:
                if count>50:
                        break
                if  rxn.IsMoleculeReactant(p[0]):
                    for b2,bb2 in enumerate(bb):
                        if count>500:
                            break
                        if  rxn.IsMoleculeReactant(bb2):
                            prodl=rxn.RunReactants((p[0], bb2))
                        else:
                            continue
                        if prodl==():
                            continue
                        prodl=prodl[0][0]
                        Chem.SanitizeMol(prodl)
                        Chem.Kekulize(prodl)
                        if not Ghose_filter(prodl):
                            continue
                        count+=1
                        prod2.append([prodl,c])
                        data[c]=[ri,p[1],b2,l]
                        products_smile[c]=Chem.MolToSmiles(prodl)
                        products_mf[c]=np.array(AllChem.GetMorganFingerprintAsBitVect(prodl,useChirality=True, radius=5, nBits = 1024))
                        c+=1
                    
        print("done",l, ": ", len(data))
        products=prod2


    return data,products_smile,products_mf,imf

if __name__ == "__main__":
    with open("mcule_bb.smi") as f:
        line = f.readlines(1000000)
    compound_list=[a.split('\t')[0] for a in line]
    filtered_cl=[]
    for mol in compound_list:
        if len(mol)<25:
            filtered_cl.append(mol)
    # filtered_cl=filtered_cl[:100]
    with open('filtered_cl.pkl', 'wb') as file:
        pickle.dump(filtered_cl, file)
    print('Start prep')
    # Save filtered_cl list as CSV
    df = pd.DataFrame(filtered_cl, columns=['Molecule'])
    df.to_csv('filtered_cl.csv', index=False)
        
    df = pd.read_excel('reactions.xls')
    reactions=df["smirks"].to_list()
    rxn=[rdChemReactions.ReactionFromSmarts(r) for r in reactions]
    bimolar = [r for r in rxn if r.GetNumReactantTemplates() == 2]
    filtered_cl=[Chem.CanonSmiles(m) for m in filtered_cl]
    bb= [Chem.MolFromSmiles(m,sanitize=True ) for m in filtered_cl]
    print('Start compute')
    a=compute_prod(bb,bimolar,run=2)
    with open('data.pkl', 'wb') as file:
        pickle.dump(a, file)
    

