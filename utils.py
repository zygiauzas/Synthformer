from rdkit import Chem

def load_enamine_building_blocks(sdf_file="Enamine_Rush-Delivery_Building_Blocks-US_251222cmpd_20250111.sdf", max_length=30):
    """
    Load building blocks from Enamine SDF file.
    
    Parameters:
    sdf_file (str): Path to the Enamine SDF file
    max_length (int): Maximum SMILES length to filter molecules
    
    Returns:
    list: List of SMILES strings for building blocks
    """
    try:
        supplier = Chem.SDMolSupplier(sdf_file)
        compound_list = []
        
        for mol in supplier:
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                if smiles and len(smiles) < max_length:
                    compound_list.append(smiles)
        
        print(f"Loaded {len(compound_list)} building blocks from {sdf_file}")
        return compound_list
    
    except FileNotFoundError:
      
        print("Error: Neither SDF nor SMI file found!")
        return []

def getlistreactants(rxn_idx, reactant_idx, precomputed_bb_masks):
    """
    Returns a list of indices of building blocks that can react with the given reaction and reactant.

    Parameters:
    rxn_idx (int): The index of the reaction (0 to n).
    reactant_idx (int): The index of the reactant position (0 or 1).
    precomputed_bb_masks (np.array): A 3D array of shape (2, num_reactions, num_building_blocks) 
                                     where masks[0] corresponds to the first reactant position,
                                     and masks[1] corresponds to the second reactant position.
    
    Returns:
    list: A list of indices (integers) representing building blocks that are compatible with the given reactant in the reaction.
    """

    # List to store the compatible building block indices
    compatible_bb_indices = []

    # Precomputed mask for the given reactant position (reactant_idx) and reaction (rxn_idx)
    mask_for_reaction = precomputed_bb_masks[reactant_idx][rxn_idx]

    # Check which building blocks are compatible by examining the mask
    num_building_blocks = mask_for_reaction.shape[0]

    # Iterate through building blocks and check if there's any match
    for bb_idx in range(num_building_blocks):
        if mask_for_reaction[bb_idx] == 1:  # If there's a match for this bb_idx
            compatible_bb_indices.append(bb_idx)

    return compatible_bb_indices
