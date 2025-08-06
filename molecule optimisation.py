import torch
import numpy as np
import pandas as pd
import pickle
import random
from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions, Descriptors
from sklearn.metrics.pairwise import cosine_similarity
from model import Transformer
import sys
import os

class SyntheticTreePipeline:
    def __init__(self, model_path='modele.pth', bb_file='Enamine_Rush-Delivery_Building_Blocks-US_251222cmpd_20250111.sdf', reactions_file='reactions.xls'):
        """
        Initialize the synthetic tree generation pipeline.
        
        Args:
            model_path: Path to the trained model
            bb_file: Path to building blocks file
            reactions_file: Path to reactions file
        """
        self.model_path = model_path
        self.bb_file = bb_file
        self.reactions_file = reactions_file
        
        # Load building blocks
        self.building_blocks = self._load_building_blocks()
        self.bb_fingerprints = self._compute_bb_fingerprints()
        
        # Load reactions
        self.reactions = self._load_reactions()
        
        # Load model
        self.model = self._load_model()
        
    def _load_building_blocks(self):
        """Load and filter building blocks from SDF file."""
        filtered_cl = []
        try:
            # Try to load from SDF file first
            supplier = Chem.SDMolSupplier(self.bb_file)
            for mol in supplier:
                if mol is not None:
                    smiles = Chem.MolToSmiles(mol)
                    if smiles and len(smiles) < 30:
                        filtered_cl.append(smiles)
        except:
            # Fallback to SMI format if SDF fails
            try:
                with open(self.bb_file) as f:
                    lines = f.readlines(1000000)
                    compound_list = [a.split('\t')[0] for a in lines]
                    for mol in compound_list:
                        if len(mol) < 30:  # Filter by length
                            filtered_cl.append(mol)
            except FileNotFoundError:
                print(f"Warning: {self.bb_file} not found. Using default building blocks.")
                filtered_cl = ['C', 'CC', 'CCC', 'CCCC', 'c1ccccc1']  # Default SMILES
        
        # Convert to RDKit molecules
        bb_mols = []
        for smiles in filtered_cl:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                bb_mols.append(mol)
        
        print(f"Loaded {len(bb_mols)} building blocks")
        return bb_mols
    
    def _compute_bb_fingerprints(self):
        """Compute Morgan fingerprints for all building blocks."""
        fingerprints = []
        for mol in self.building_blocks:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fp_array = np.array(list(fp.ToBitString())).astype(np.float32)
            fingerprints.append(fp_array)
        return np.array(fingerprints)
    
    def _load_reactions(self):
        """Load reaction rules from Excel file."""
        try:
            df = pd.read_excel(self.reactions_file)
            reactions_smirks = df["smirks"].to_list()
            rxn_objects = []
            for smirks in reactions_smirks:
                try:
                    rxn = rdChemReactions.ReactionFromSmarts(smirks)
                    if rxn is not None and rxn.GetNumReactantTemplates() == 2:
                        rxn_objects.append(rxn)
                except:
                    continue
            print(f"Loaded {len(rxn_objects)} bimolecular reactions")
            return rxn_objects
        except FileNotFoundError:
            print(f"Warning: {self.reactions_file} not found. Using default reactions.")
            # Default Suzuki coupling reaction
            default_smirks = ["[c:1][Br:2].[c:3][B:4]([OH])([OH])>>[c:1][c:3]"]
            return [rdChemReactions.ReactionFromSmarts(smirks) for smirks in default_smirks]
    
    def _load_model(self):
        """Load the trained transformer model."""
        try:
            bbvocablen = len(self.building_blocks)
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
            model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
            model.eval()
            print("Model loaded successfully")
            return model
        except FileNotFoundError:
            print(f"Warning: {self.model_path} not found. Model functionality disabled.")
            return None
    
    def find_nearest_building_block(self, target_mol):
        """
        Find the nearest building block based on Morgan fingerprint similarity.
        
        Args:
            target_mol: RDKit molecule object
            
        Returns:
            Index of the nearest building block
        """
        if target_mol is None:
            return random.randint(0, len(self.building_blocks) - 1)
        
        # Compute fingerprint for target molecule
        target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, 2, nBits=1024)
        target_array = np.array(list(target_fp.ToBitString())).astype(np.float32)
        
        # Compute similarities
        similarities = cosine_similarity([target_array], self.bb_fingerprints)[0]
        
        # Return index of most similar building block
        return np.argmax(similarities)
    
    def apply_reaction_rules(self, mol1, mol2):
        """
        Apply reaction rules to two molecules to generate new molecules.
        
        Args:
            mol1: First reactant molecule
            mol2: Second reactant molecule
            
        Returns:
            List of product molecules
        """
        products = []
        
        for rxn in self.reactions:
            try:
                rxn.Initialize()
                if rxn.IsMoleculeReactant(mol1) and rxn.IsMoleculeReactant(mol2):
                    prods = rxn.RunReactants((mol1, mol2))
                    for prod_set in prods:
                        for prod in prod_set:
                            try:
                                Chem.SanitizeMol(prod)
                                # Convert to SMILES and back to ensure validity
                                smiles = Chem.MolToSmiles(prod)
                                clean_mol = Chem.MolFromSmiles(smiles)
                                if clean_mol is not None:
                                    products.append(clean_mol)
                            except:
                                continue
            except Exception as e:
                continue
        
        return products
    
    def score_molecule(self, mol):
        """
        Score a molecule based on various properties.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Float score (higher is better)
        """
        if mol is None:
            return 0.0
        
        try:
            # Compute various molecular descriptors
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            
            # Lipinski's rule of five compliance
            lipinski_score = 0
            if mw <= 500: lipinski_score += 1
            if logp <= 5: lipinski_score += 1
            if hbd <= 5: lipinski_score += 1
            if hba <= 10: lipinski_score += 1
            
            # Penalize too many rotatable bonds
            flexibility_penalty = max(0, 1 - (rotatable_bonds - 5) * 0.1)
            
            # Combined score (normalize to 0-1)
            score = (lipinski_score / 4.0) * flexibility_penalty
            
            return score
            
        except Exception as e:
            return 0.0
    
    def generate_synthetic_tree(self, initial_molecules=None, T=3, k=10):
        """
        Generate synthetic tree following the algorithm.
        
        Args:
            initial_molecules: Initial set of molecules (if None, uses building blocks)
            T: Number of iterations
            k: Top-k selection size
            
        Returns:
            Set of molecules in the synthetic tree
        """
        print(f"Starting synthetic tree generation with T={T}, k={k}")
        
        # Initialize
        synthetic_tree = set()
        
        if initial_molecules is None:
            current_molecules = self.building_blocks.copy()
        else:
            current_molecules = initial_molecules.copy()
        
        print(f"Starting with {len(current_molecules)} molecules")
        
        for t in range(T):
            print(f"\nIteration {t+1}/{T}")
            new_molecules = []
            
            for i, mol in enumerate(current_molecules):
                if i % 10 == 0:
                    print(f"  Processing molecule {i+1}/{len(current_molecules)}")
                
                for bb in self.building_blocks:
                    # Find nearest building block
                    nearest_idx = self.find_nearest_building_block(bb)
                    nearest_bb = self.building_blocks[nearest_idx]
                    
                    # Apply reaction rules
                    products = self.apply_reaction_rules(mol, nearest_bb)
                    new_molecules.extend(products)
            
            print(f"  Generated {len(new_molecules)} new molecules")
            
            if not new_molecules:
                print("  No new molecules generated, stopping")
                break
            
            # Score all new molecules
            scored_molecules = []
            for mol in new_molecules:
                score = self.score_molecule(mol)
                scored_molecules.append((mol, score))
            
            # Select top-k molecules
            scored_molecules.sort(key=lambda x: x[1], reverse=True)
            top_k = scored_molecules[:k]
            current_molecules = [mol for mol, score in top_k]
            
            print(f"  Selected top {len(current_molecules)} molecules")
            print(f"  Score range: {top_k[-1][1]:.3f} - {top_k[0][1]:.3f}")
            
            # Update synthetic tree
            synthetic_tree.update(current_molecules)
        
        print(f"\nSynthetic tree generation complete. Total molecules: {len(synthetic_tree)}")
        return synthetic_tree
    
    def save_results(self, molecules, filename='synthetic_tree_results.pkl'):
        """Save the generated molecules to a file."""
        # Convert molecules to SMILES for storage
        smiles_list = []
        for mol in molecules:
            try:
                smiles = Chem.MolToSmiles(mol)
                smiles_list.append(smiles)
            except:
                continue
        
        with open(filename, 'wb') as f:
            pickle.dump(smiles_list, f)
        
        print(f"Results saved to {filename}")
        return smiles_list

def main():
    """Main function to run the synthetic tree generation pipeline."""
    # Initialize pipeline
    pipeline = SyntheticTreePipeline()
    
    # Generate synthetic tree
    synthetic_tree = pipeline.generate_synthetic_tree(T=3, k=10)
    
    # Save results
    smiles_results = pipeline.save_results(synthetic_tree)
    
    # Print some results
    print("\nSample generated molecules:")
    for i, smiles in enumerate(smiles_results[:10]):
        print(f"{i+1}. {smiles}")

if __name__ == "__main__":
    main() 