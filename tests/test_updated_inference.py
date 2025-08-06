#!/usr/bin/env python3
"""
Test script to verify all updated inference functions work correctly
"""

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Transformer

def test_model_predict():
    """Test the updated model.predict method"""
    print("Testing updated model.predict method...")
    
    device = torch.device("cpu")  # Use CPU for testing
    
    # Create model
    model = Transformer(
        source_vocab_size=50,
        target_vocab_size=100,
        embedding_dim=256,
        source_max_seq_len=64,
        target_max_seq_len=64,
        num_layers=2,
        num_heads=4,
        dropout=0.1
    )
    model.to(device)
    model.eval()
    
    # Create test data
    batch_size = 1
    seq_len = 4
    p4 = torch.randn(batch_size, seq_len, 11, device=device)
    mflist = torch.randn(batch_size, seq_len, 1024, device=device)
    
    # Create fingerprint list
    fingerprint_list = []
    for i in range(10):
        fp = np.random.randint(0, 2, 1024).astype(np.float32)
        fingerprint_list.append(fp)
    
    # Test predict method
    with torch.no_grad():
        bb_pred, bb_fingerprint, reaction_pred = model.predict(
            p4, mflist, fingerprint_list, end_token_id=99
        )
    
    print(f"✅ model.predict() works")
    print(f"   Building block prediction: {bb_pred}")
    print(f"   Building block fingerprint shape: {bb_fingerprint.shape if bb_fingerprint is not None else None}")
    print(f"   Reaction prediction: {reaction_pred}")
    
    return True

def test_autoregressive_imports():
    """Test that autoregressive functions can be imported"""
    print("\nTesting autoregressive function imports...")
    
    try:
        from inference import process_reactions_autoregressive
        print("✅ process_reactions_autoregressive imported from inference.py")
    except ImportError as e:
        print(f"❌ Failed to import from inference.py: {e}")
        return False
    
    try:
        from inference_pdb_bind import process_reactions_autoregressive
        print("✅ process_reactions_autoregressive imported from inference_pdb_bind.py")
    except ImportError as e:
        print(f"❌ Failed to import from inference_pdb_bind.py: {e}")
        return False
    
    try:
        from inference_hit_expansion import process_reactions_autoregressive_hit_expansion
        print("✅ process_reactions_autoregressive_hit_expansion imported from inference_hit_expansion.py")
    except ImportError as e:
        print(f"❌ Failed to import from inference_hit_expansion.py: {e}")
        return False
    
    return True

def test_backward_compatibility():
    """Test that old inference functions still work"""
    print("\nTesting backward compatibility...")
    
    try:
        from inference import process_reactions
        from inference_pdb_bind import process_reactions
        from inference_hit_expansion import process_reactions
        print("✅ Original process_reactions functions still available")
        return True
    except ImportError as e:
        print(f"❌ Backward compatibility broken: {e}")
        return False

def test_loss_function():
    """Test the new loss function"""
    print("\nTesting new loss function...")
    
    try:
        from train import calculate_building_block_loss
        
        # Create test data
        device = torch.device("cpu")
        batch_size = 2
        seq_len = 4
        embedding_dim = 256
        
        bb_representations = torch.randn(batch_size, seq_len, embedding_dim, device=device)
        buildingblock = torch.randint(0, 10, (batch_size, seq_len), device=device)
        
        # Create fingerprint list
        fingerprint_list = []
        for i in range(10):
            fp = np.random.randint(0, 2, 1024).astype(np.float32)
            fingerprint_list.append(fp)
        
        # Create model with fingerprint encoder
        model = Transformer(
            source_vocab_size=50,
            target_vocab_size=100,
            embedding_dim=embedding_dim,
            source_max_seq_len=64,
            target_max_seq_len=64,
            num_layers=2,
            num_heads=4,
            dropout=0.1
        )
        model.to(device)
        
        # Test loss calculation
        loss = calculate_building_block_loss(
            bb_representations, buildingblock, fingerprint_list, model, device
        )
        
        print(f"✅ Building block loss calculation works: {loss.item():.4f}")
        return True
        
    except Exception as e:
        print(f"❌ Loss function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests"""
    print("="*50)
    print("TESTING UPDATED INFERENCE SYSTEM")
    print("="*50)
    
    tests = [
        test_model_predict,
        test_autoregressive_imports,
        test_backward_compatibility,
        test_loss_function
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("="*50)
    
    if passed == total:
        print("🎉 All tests passed! The inference system is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 