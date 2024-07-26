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





def precompute_bb_masks(ctx,building_blocks):
    

    print("Precomputing building blocks masks for each reaction and reactant position...")
    
    masks = np.zeros((2, len(ctx), len(building_blocks)))
    print(masks.shape)
    for rxn_i,rxn in tqdm(enumerate(ctx)):
        reaction = rxn
        reactants = reaction.GetReactants()
        for bb_j, bb in enumerate(building_blocks):
            if bb.HasSubstructMatch(reactants[0]):
                masks[0, rxn_i, bb_j] = 1
            if bb.HasSubstructMatch(reactants[1]):
                masks[1, rxn_i, bb_j] = 1

    print(f"Saving precomputed masks to of shape={masks.shape} to {PATH}")
    with open(PATH, "wb") as f:
        pickle.dump(masks, f)


if __name__ == "__main__":
    PATH =  "precomputed_bb_masks.pkl"
    with open("mcule_bb.smi") as f:
        line = f.readlines(1000000)
    compound_list=[a.split('\t')[0] for a in line][:200]
    
    df = pd.read_excel('reactions.xls')
    reactions=df["smirks"].to_list()
    rxn=[rdChemReactions.ReactionFromSmarts(r) for r in reactions]
    bimolar = [r for r in rxn if r.GetNumReactantTemplates() == 2]
    bb= [Chem.MolFromSmiles(m) for m in compound_list]
    precompute_bb_masks(bimolar ,bb)
    print("Done!")
    with open(PATH, "rb") as f:
        masks = pickle.load(f)
    print(masks.shape)

    