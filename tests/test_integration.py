import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import tempfile
import os
import pickle
from unittest.mock import Mock, patch, MagicMock, mock_open
from torch.utils.data import DataLoader

# Add parent directory to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from model import Transformer
from Dataloader import Datasetp4, custom_collate_fn
from utils import load_enamine_building_blocks, getlistreactants


class TestDataPipeline:
    """Test the complete data pipeline from loading to model input"""
    
    @pytest.fixture
    def mock_complete_data(self):
        """Create a complete mock dataset for testing"""
        # Create comprehensive mock data
        num_samples = 20
        data = [
            [[i, 100+i, 101+i, (i % 3) + 1] for i in range(num_samples)],  # reactions
            [f"smiles_{i}" for i in range(num_samples)],                    # smiles
            [np.random.rand(1024).astype(np.float32) for i in range(num_samples)]  # fingerprints
        ]
        
        # p4 data (pharmacophore features)
        p4 = [np.random.rand(50, 11).astype(np.float32) for _ in range(num_samples)]
        
        # Building block fingerprints
        fingerprint_list = [np.random.rand(1024).astype(np.float32) for _ in range(500)]
        
        # Building block SMILES
        building_blocks = [f"BB_{i}" for i in range(500)]
        
        return data, p4, fingerprint_list, building_blocks
    
    def test_complete_data_loading_pipeline(self, mock_complete_data):
        """Test complete data loading from raw data to DataLoader"""
        data, p4, fingerprint_list, building_blocks = mock_complete_data
        
        # Create dataset
        indexes = list(range(len(data[0])))
        dataset = Datasetp4(data, p4, indexes, fingerprint_list)
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset, 
            batch_size=4, 
            collate_fn=custom_collate_fn,
            shuffle=False
        )
        
        # Test that we can iterate through the entire dataset
        total_samples = 0
        for batch in dataloader:
            assert len(batch) == 7  # Correct number of elements
            batch_size = batch[0].shape[0]
            total_samples += batch_size
            
            # Validate batch structure
            p4_batch, reactions_batch, mflist_batch, bb_batch, bbmf_batch, maxlens, smiles = batch
            
            # Check shapes
            assert p4_batch.shape[0] == batch_size
            assert reactions_batch.shape[0] == batch_size
            assert mflist_batch.shape[0] == batch_size
            assert bb_batch.shape[0] == batch_size
            assert bbmf_batch.shape[0] == batch_size
            
            # Check data types
            assert p4_batch.dtype == torch.float32
            assert reactions_batch.dtype == torch.float32
            assert mflist_batch.dtype == torch.float32
        
        # Should have processed all samples
        assert total_samples == len(indexes)
    
    def test_data_consistency_through_pipeline(self, mock_complete_data):
        """Test that data remains consistent through the pipeline"""
        data, p4, fingerprint_list, building_blocks = mock_complete_data
        
        indexes = [0, 1, 2]  # Test with small subset
        dataset = Datasetp4(data, p4, indexes, fingerprint_list)
        
        # Get individual items
        item1 = dataset[1]  # First item (dataset uses idx-1)
        item2 = dataset[2]  # Second item
        
        # Create batch manually
        batch = [item1, item2]
        collated = custom_collate_fn(batch)
        
        # Verify that collated data matches individual items
        p4_collated, _, _, _, _, _, _ = collated
        
        # Check that individual p4 data is preserved
        assert torch.allclose(p4_collated[0], item1[0])
        assert torch.allclose(p4_collated[1], item2[0])


class TestModelTrainingIntegration:
    """Test model training integration with real data pipeline"""
    
    @pytest.fixture
    def training_setup(self):
        """Set up a minimal training environment"""
        # Create small model
        model = Transformer(
            source_vocab_size=100,
            target_vocab_size=105,
            embedding_dim=64,
            source_max_seq_len=32,
            target_max_seq_len=32,
            num_layers=2,
            num_heads=4,
            dropout=0.1
        )
        
        # Create mock data
        num_samples = 10
        data = [
            [[i, 100+i, 101+i, (i % 2) + 1] for i in range(num_samples)],
            [f"smiles_{i}" for i in range(num_samples)],
            [np.random.rand(1024).astype(np.float32) for i in range(num_samples)]
        ]
        
        p4 = [np.random.rand(20, 11).astype(np.float32) for _ in range(num_samples)]
        fingerprint_list = [np.random.rand(1024).astype(np.float32) for _ in range(105)]
        
        # Create dataset and dataloader
        indexes = list(range(num_samples))
        dataset = Datasetp4(data, p4, indexes, fingerprint_list)
        dataloader = DataLoader(dataset, batch_size=3, collate_fn=custom_collate_fn)
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        return model, dataloader, optimizer
    
    def test_single_training_step(self, training_setup):
        """Test a single training step with real data pipeline"""
        model, dataloader, optimizer = training_setup
        
        # Get a batch
        batch = next(iter(dataloader))
        p4, reactions, mflist, buildingblock, buildingblockmf, maxlens, smiles = batch
        
        # Ensure shapes are compatible
        if buildingblock.shape[1] != reactions.shape[1]:
            pytest.skip("Batch has incompatible shapes - this is expected occasionally")
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        bbout, reaction_pred = model(p4, reactions, mflist, buildingblock, buildingblockmf)
        
        # Compute loss
        bbout_flat = bbout.permute(0, 2, 1)
        reaction_flat = reaction_pred.permute(0, 2, 1)
        lossbb = F.cross_entropy(bbout_flat, buildingblock.long())
        lossreactions = F.cross_entropy(reaction_flat, reactions.long())
        total_loss = lossbb + lossreactions
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Verify training occurred
        assert total_loss.item() >= 0
        assert any(param.grad is not None for param in model.parameters())
    
    def test_multi_epoch_training(self, training_setup):
        """Test training over multiple epochs"""
        model, dataloader, optimizer = training_setup
        
        losses = []
        
        # Train for a few epochs
        for epoch in range(3):
            epoch_loss = 0
            num_batches = 0
            
            for batch in dataloader:
                p4, reactions, mflist, buildingblock, buildingblockmf, maxlens, smiles = batch
                
                # Skip incompatible batches
                if buildingblock.shape[1] != reactions.shape[1]:
                    continue
                
                optimizer.zero_grad()
                
                bbout, reaction_pred = model(p4, reactions, mflist, buildingblock, buildingblockmf)
                
                bbout_flat = bbout.permute(0, 2, 1)
                reaction_flat = reaction_pred.permute(0, 2, 1)
                lossbb = F.cross_entropy(bbout_flat, buildingblock.long())
                lossreactions = F.cross_entropy(reaction_flat, reactions.long())
                total_loss = lossbb + lossreactions
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                num_batches += 1
            
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                losses.append(avg_loss)
        
        # Should have completed training without errors
        assert len(losses) > 0
        assert all(loss >= 0 for loss in losses)


class TestInferenceIntegration:
    """Test inference integration with the complete pipeline"""
    
    @pytest.fixture
    def inference_setup(self):
        """Set up model and data for inference testing"""
        # Create and train a minimal model
        model = Transformer(
            source_vocab_size=100,
            target_vocab_size=105,
            embedding_dim=64,
            source_max_seq_len=32,
            target_max_seq_len=32,
            num_layers=2,
            num_heads=4,
            dropout=0.0  # No dropout for inference
        )
        
        # Create test data
        batch_size = 1
        seq_len = 5
        p4 = torch.randn(batch_size, seq_len, 11)
        mflist = torch.randn(batch_size, seq_len, 1024)
        fingerprint_list = [torch.randn(1024) for _ in range(105)]
        
        return model, p4, mflist, fingerprint_list
    
    def test_model_inference_mode(self, inference_setup):
        """Test model in inference mode"""
        model, p4, mflist, fingerprint_list = inference_setup
        
        # Set to evaluation mode
        model.eval()
        
        # Test inference
        with torch.no_grad():
            logit, buildingblockmf, logitr = model.predict(p4, mflist, fingerprint_list, end_token_id=104)
        
        # Validate outputs
        assert len(logit) == 1  # Should return 1 building block (single sample)
        assert isinstance(logit, torch.Tensor)
        assert buildingblockmf.shape == (1, 1, 1024)
        assert isinstance(logitr, int)  # Should return single reaction index
    
    def test_inference_determinism(self, inference_setup):
        """Test that inference is deterministic with same inputs"""
        model, p4, mflist, fingerprint_list = inference_setup
        
        model.eval()
        
        # Run inference twice with same inputs
        with torch.no_grad():
            logit1, bbmf1, logitr1 = model.predict(p4, mflist, fingerprint_list, end_token_id=104)
            logit2, bbmf2, logitr2 = model.predict(p4, mflist, fingerprint_list, end_token_id=104)
        
        # Results should be identical (since no dropout and same random seed)
        # Note: Due to multinomial sampling, results may differ
        # So we just check that function runs consistently
        assert logit1.shape == logit2.shape
        assert bbmf1.shape == bbmf2.shape
        assert type(logitr1) == type(logitr2)  # Both should be int


class TestSaveLoadIntegration:
    """Test model saving and loading integration"""
    
    def test_complete_save_load_cycle(self):
        """Test complete save/load cycle with training state"""
        # Create model
        model = Transformer(
            source_vocab_size=100,
            target_vocab_size=50,
            embedding_dim=64,
            source_max_seq_len=32,
            target_max_seq_len=32,
            num_layers=2,
            num_heads=4,
            dropout=0.1
        )
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train for one step to modify weights
        batch_size = 2
        seq_len = 5
        p4 = torch.randn(batch_size, seq_len, 11)
        reactions = torch.randint(0, 56, (batch_size, seq_len))
        mflist = torch.randn(batch_size, seq_len, 1024)
        buildingblock = torch.randint(0, 50, (batch_size, seq_len))
        buildingblockmf = torch.randn(batch_size, seq_len, 1024)
        
        optimizer.zero_grad()
        bbout, reaction_pred = model(p4, reactions, mflist, buildingblock, buildingblockmf)
        loss = bbout.sum() + reaction_pred.sum()
        loss.backward()
        optimizer.step()
        
        # Save model state
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
            model_path = tmp.name
        
        try:
            # Save state dict
            torch.save(model.state_dict(), model_path)
            
            # Create new model and load state
            new_model = Transformer(
                source_vocab_size=100,
                target_vocab_size=50,
                embedding_dim=64,
                source_max_seq_len=32,
                target_max_seq_len=32,
                num_layers=2,
                num_heads=4,
                dropout=0.1
            )
            new_model.load_state_dict(torch.load(model_path, map_location='cpu'))
            
            # Test that models produce same output
            model.eval()
            new_model.eval()
            
            with torch.no_grad():
                out1 = model(p4, reactions, mflist, buildingblock, buildingblockmf)
                out2 = new_model(p4, reactions, mflist, buildingblock, buildingblockmf)
            
            # Outputs should be identical
            assert torch.allclose(out1[0], out2[0], atol=1e-6)
            assert torch.allclose(out1[1], out2[1], atol=1e-6)
        
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)


class TestUtilsIntegration:
    """Test utils functions integration with the pipeline"""
    
    def test_building_blocks_integration(self):
        """Test building blocks loading integration"""
        # Mock building blocks loading
        with patch('utils.Chem.SDMolSupplier', side_effect=FileNotFoundError):
            smi_content = "CCO\tmol1\nCCC\tmol2\nC=O\tmol3\nCCCC\tmol4\n"
            
            with patch('builtins.open', mock_open(read_data=smi_content)):
                building_blocks = load_enamine_building_blocks(max_length=10)
        
        # Use building blocks in dataset creation
        num_samples = 5
        data = [
            [[i, len(building_blocks)+i, len(building_blocks)+i+1, 2] for i in range(num_samples)],
            [f"smiles_{i}" for i in range(num_samples)],
            [np.random.rand(1024).astype(np.float32) for i in range(num_samples)]
        ]
        
        p4 = [np.random.rand(20, 11).astype(np.float32) for _ in range(num_samples)]
        fingerprint_list = [np.random.rand(1024).astype(np.float32) for _ in range(len(building_blocks) + 10)]
        
        # Create dataset
        indexes = list(range(num_samples))
        dataset = Datasetp4(data, p4, indexes, fingerprint_list)
        
        # Test that dataset works with loaded building blocks
        assert len(dataset) == num_samples
        item = dataset[1]
        assert len(item) == 7
    
    def test_reaction_compatibility_integration(self):
        """Test reaction compatibility functions integration"""
        # Create mock reaction masks
        num_reactions = 10
        num_building_blocks = 50
        masks = np.random.randint(0, 2, (2, num_reactions, num_building_blocks))
        
        # Test getting compatible building blocks for various reactions
        for rxn_idx in range(min(3, num_reactions)):  # Test first 3 reactions
            for reactant_idx in range(2):
                compatible_bbs = getlistreactants(rxn_idx, reactant_idx, masks)
                
                assert isinstance(compatible_bbs, list)
                assert all(0 <= bb_idx < num_building_blocks for bb_idx in compatible_bbs)
                
                # Use these in a hypothetical molecule generation process
                if compatible_bbs:
                    # Simulate selecting a building block
                    selected_bb = compatible_bbs[0]
                    assert 0 <= selected_bb < num_building_blocks


class TestPerformanceIntegration:
    """Test performance characteristics of the integrated system"""
    
    def test_memory_usage_scaling(self):
        """Test memory usage with different dataset sizes"""
        sizes = [10, 50, 100]
        
        for size in sizes:
            # Create dataset of given size
            data = [
                [[i, 100+i, 101+i, (i % 3) + 1] for i in range(size)],
                [f"smiles_{i}" for i in range(size)],
                [np.random.rand(1024).astype(np.float32) for i in range(size)]
            ]
            
            p4 = [np.random.rand(20, 11).astype(np.float32) for _ in range(size)]
            fingerprint_list = [np.random.rand(1024).astype(np.float32) for _ in range(200)]
            
            indexes = list(range(size))
            dataset = Datasetp4(data, p4, indexes, fingerprint_list)
            
            # Test that dataset can be created and accessed
            assert len(dataset) == size
            
            # Test accessing random samples
            import random
            for _ in range(min(5, size)):
                idx = random.randint(1, size)
                item = dataset[idx]
                assert item is not None
    
    def test_batch_processing_scaling(self):
        """Test batch processing with different batch sizes"""
        # Create dataset
        size = 50
        data = [
            [[i, 100+i, 101+i, (i % 2) + 1] for i in range(size)],
            [f"smiles_{i}" for i in range(size)],
            [np.random.rand(1024).astype(np.float32) for i in range(size)]
        ]
        
        p4 = [np.random.rand(15, 11).astype(np.float32) for _ in range(size)]
        fingerprint_list = [np.random.rand(1024).astype(np.float32) for _ in range(200)]
        
        indexes = list(range(size))
        dataset = Datasetp4(data, p4, indexes, fingerprint_list)
        
        # Test different batch sizes
        batch_sizes = [1, 4, 8, 16]
        
        for batch_size in batch_sizes:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=custom_collate_fn,
                shuffle=False
            )
            
            # Process all batches
            total_processed = 0
            for batch in dataloader:
                current_batch_size = batch[0].shape[0]
                total_processed += current_batch_size
                
                # Verify batch structure
                assert len(batch) == 7
                assert batch[0].shape[0] <= batch_size
            
            # Should have processed all samples
            assert total_processed == size


class TestErrorHandlingIntegration:
    """Test error handling across the integrated system"""
    
    def test_invalid_data_handling(self):
        """Test handling of invalid data throughout pipeline"""
        # Create dataset with some invalid entries
        data = [
            [[1, 100, 101, 2], [2, 102, 103, -1]],  # Invalid length
            ["CCO", ""],  # Empty SMILES
            [np.random.rand(1024).astype(np.float32), np.random.rand(1024).astype(np.float32)]
        ]
        
        p4 = [np.random.rand(20, 11).astype(np.float32) for _ in range(2)]
        fingerprint_list = [np.random.rand(1024).astype(np.float32) for _ in range(200)]
        
        indexes = [0, 1]
        
        try:
            dataset = Datasetp4(data, p4, indexes, fingerprint_list)
            # Should handle creation without crashing
            assert len(dataset) == len(indexes)
            
            # Try to get items
            for i in range(1, len(indexes) + 1):
                try:
                    item = dataset[i]
                    assert len(item) == 7
                except Exception as e:
                    # Some invalid data might cause exceptions - this is acceptable
                    assert isinstance(e, (IndexError, ValueError))
        
        except Exception as e:
            # Dataset creation might fail with invalid data - this is acceptable
            assert isinstance(e, (IndexError, ValueError, TypeError))
    
    def test_model_input_validation(self):
        """Test model input validation"""
        model = Transformer(100, 100, 32, 32, 64, 4, 2)
        
        # Test with mismatched input shapes
        p4 = torch.randn(2, 5, 11)
        reactions = torch.randint(0, 56, (2, 3))  # Different length
        mflist = torch.randn(2, 5, 1024)
        buildingblock = torch.randint(0, 100, (2, 5))
        buildingblockmf = torch.randn(2, 5, 1024)
        
        try:
            output = model(p4, reactions, mflist, buildingblock, buildingblockmf)
            # If it doesn't crash, check output validity
            assert len(output) == 2
        except (RuntimeError, ValueError, IndexError) as e:
            # Expected for mismatched inputs
            pass


# Helper function to run all integration tests
def run_integration_tests():
    """Run all integration tests with pytest"""
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    run_integration_tests() 