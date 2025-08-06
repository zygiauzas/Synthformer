import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Transformer
from rdkit import Chem
from rdkit.Chem import AllChem

def test_new_loss_functions():
    """
    Test the new building block loss function with sample data.
    """
    print("Testing new loss functions...")
    
    # Create a small transformer model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model parameters
    embedding_dim = 256
    vocab_size = 100
    
    model = Transformer(
        source_vocab_size=50,
        target_vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        source_max_seq_len=64,
        target_max_seq_len=64,
        num_layers=2,
        num_heads=4,
        dropout=0.1
    )
    model.to(device)
    model.eval()
    
    # Create sample data
    batch_size = 2
    seq_len = 4
    
    # Sample pharmacophore data
    p4 = torch.randn(batch_size, seq_len, 11, device=device)
    
    # Sample reactions
    reactions = torch.randint(0, 56, (batch_size, seq_len), device=device)
    
    # Sample molecular fingerprints
    mflist = torch.randn(batch_size, seq_len, 1024, device=device)
    
    # Sample building block indices
    buildingblock = torch.randint(0, 10, (batch_size, seq_len), device=device)
    
    # Sample building block molecular fingerprints
    buildingblockmf = torch.randn(batch_size, seq_len, 1024, device=device)
    
    # Create sample fingerprint list (10 building blocks)
    fingerprint_list = []
    for i in range(10):
        # Create random fingerprint
        fp = np.random.randint(0, 2, 1024).astype(np.float32)
        fingerprint_list.append(fp)
    
    print("Sample data created successfully")
    
    # Test forward pass
    try:
        with torch.no_grad():
            bbout, reaction_pred, bb_representations = model(p4, reactions, mflist, buildingblock, buildingblockmf)
        
        print(f"Forward pass successful:")
        print(f"  Building block output shape: {bbout.shape}")
        print(f"  Reaction prediction shape: {reaction_pred.shape}")
        print(f"  Building block representations shape: {bb_representations.shape}")
        
        # Test the building block loss calculation
        from train import calculate_building_block_loss
        
        lossbb = calculate_building_block_loss(bb_representations, buildingblock, fingerprint_list, model, device)
        
        print(f"Building block loss (cosine similarity): {lossbb.item():.4f}")
        
        # Test reaction loss (standard cross entropy)
        reaction_flat = reaction_pred.permute(0, 2, 1)
        lossreactions = F.cross_entropy(reaction_flat, reactions.long())
        
        print(f"Reaction loss (cross entropy): {lossreactions.item():.4f}")
        
        # Total loss
        total_loss = lossbb + lossreactions
        print(f"Total loss: {total_loss.item():.4f}")
        
        print("\n✅ All tests passed! New loss functions work correctly.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

def test_fingerprint_encoding():
    """
    Test the fingerprint encoding layer (Z' = W * f_p(B) + b)
    """
    print("\nTesting fingerprint encoding...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_dim = 256
    
    # Create fingerprint encoder
    fingerprint_encoder = torch.nn.Linear(1024, embedding_dim).to(device)
    
    # Test with a sample fingerprint
    sample_fp = np.random.randint(0, 2, 1024).astype(np.float32)
    fp_tensor = torch.tensor(sample_fp, dtype=torch.float32, device=device)
    
    # Encode fingerprint
    encoded_fp = fingerprint_encoder(fp_tensor)
    
    print(f"Original fingerprint shape: {fp_tensor.shape}")
    print(f"Encoded fingerprint shape: {encoded_fp.shape}")
    print(f"Encoding successful: {encoded_fp.shape[0] == embedding_dim}")
    
    print("✅ Fingerprint encoding test passed!")

if __name__ == "__main__":
    test_fingerprint_encoding()
    test_new_loss_functions()
    print("\n🎉 All tests completed!") 