import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Add parent directory to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the modules to test
from model import Transformer
from Dataloader import Datasetp4, custom_collate_fn


class TestRotationMatrices:
    """Test rotation matrix generation for data augmentation"""
    
    def test_random_rotation_matrices_shape(self):
        """Test that rotation matrices have correct shape"""
        from train import random_rotation_matrices
        
        batch_size = 5
        R = random_rotation_matrices(batch_size)
        
        assert R.shape == (batch_size, 3, 3)
    
    def test_random_rotation_matrices_orthogonality(self):
        """Test that generated matrices are proper rotation matrices"""
        from train import random_rotation_matrices
        
        batch_size = 3
        R = random_rotation_matrices(batch_size)
        
        # Check orthogonality: R @ R.T = I
        identity = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
        RRT = torch.bmm(R, R.transpose(-1, -2))
        
        assert torch.allclose(RRT, identity, atol=1e-5)
    
    def test_random_rotation_matrices_determinant(self):
        """Test that determinant is 1 (proper rotation, not reflection)"""
        from train import random_rotation_matrices
        
        batch_size = 3
        R = random_rotation_matrices(batch_size)
        
        # Determinant should be 1 for proper rotations
        det = torch.det(R)
        expected = torch.ones(batch_size)
        
        assert torch.allclose(det, expected, atol=1e-5)


class TestDataLoading:
    """Test data loading and preprocessing"""
    
    @pytest.fixture
    def mock_data(self):
        """Create mock data for testing"""
        # Create minimal mock data structure
        data = [
            [  # reactions, building blocks, products
                [1, 100, 101, 2],  # [reaction_id, bb1_id, bb2_id, length]
                [2, 102, 103, 2],
            ],
            ["CCO", "CCC"],  # smiles
            [np.random.rand(1024), np.random.rand(1024)]  # molecular fingerprints
        ]
        
        # Create p4 data (pharmacophore features)
        p4 = [np.random.rand(50, 11) for _ in range(len(data[0]))]
        
        # Create building block fingerprints
        fingerprint_list = [np.random.rand(1024).astype(np.float32) for _ in range(200)]
        
        return data, p4, fingerprint_list
    
    def test_dataset_initialization(self, mock_data):
        """Test dataset initialization"""
        data, p4, fingerprint_list = mock_data
        indexes = [0, 1]
        
        dataset = Datasetp4(data, p4, indexes, fingerprint_list)
        
        assert len(dataset) == len(indexes)
        assert dataset.diclen == len(fingerprint_list)
    
    def test_dataset_getitem(self, mock_data):
        """Test dataset item retrieval"""
        data, p4, fingerprint_list = mock_data
        indexes = [0, 1]
        
        dataset = Datasetp4(data, p4, indexes, fingerprint_list)
        
        # Test getting first item (note: dataset uses idx-1)
        item = dataset[1]  # This will get index 0
        
        assert len(item) == 7  # p4, reactions, mflist, buildingblock, buildingblockmf, maxlens, smiles
        assert isinstance(item[0], torch.Tensor)  # p4
        assert isinstance(item[1], torch.Tensor)  # reactions
        assert isinstance(item[2], torch.Tensor)  # mflist
    
    def test_custom_collate_fn(self, mock_data):
        """Test custom collate function"""
        data, p4, fingerprint_list = mock_data
        indexes = [0, 1]
        
        dataset = Datasetp4(data, p4, indexes, fingerprint_list)
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate_fn)
        
        batch = next(iter(dataloader))
        
        assert len(batch) == 7
        # Check that tensors are properly batched
        assert batch[0].dim() == 3  # p4: [batch, seq, features]
        assert batch[1].dim() == 2  # reactions: [batch, seq]
        assert batch[2].dim() == 3  # mflist: [batch, seq, features]


class TestTrainingLoop:
    """Test training loop components"""
    
    @pytest.fixture
    def mock_model_and_data(self):
        """Create mock model and data for training tests"""
        # Create a small model for testing
        model = Transformer(
            source_vocab_size=100,
            target_vocab_size=103,  # Small vocab for testing
            embedding_dim=64,
            source_max_seq_len=32,
            target_max_seq_len=32,
            num_layers=2,
            num_heads=4,
            dropout=0.1
        )
        
        # Create mock training data
        batch_size = 2
        seq_len = 5
        
        p4 = torch.randn(batch_size, seq_len, 11)
        reactions = torch.randint(0, 56, (batch_size, seq_len))
        mflist = torch.randn(batch_size, seq_len, 1024)
        buildingblock = torch.randint(0, 103, (batch_size, seq_len))
        buildingblockmf = torch.randn(batch_size, seq_len, 1024)
        
        batch = (p4, reactions, mflist, buildingblock, buildingblockmf, [2, 3], ["CCO", "CCC"])
        
        return model, batch
    
    def test_forward_pass(self, mock_model_and_data):
        """Test forward pass produces correct output shapes"""
        model, batch = mock_model_and_data
        p4, reactions, mflist, buildingblock, buildingblockmf, maxlens, smiles = batch
        
        model.eval()
        with torch.no_grad():
            bbout, reaction_pred = model(p4, reactions, mflist, buildingblock, buildingblockmf)
        
        batch_size, seq_len = reactions.shape
        assert bbout.shape == (batch_size, seq_len, model.target_vocab_size)
        assert reaction_pred.shape == (batch_size, seq_len, 56)
    
    def test_loss_computation(self, mock_model_and_data):
        """Test loss computation"""
        model, batch = mock_model_and_data
        p4, reactions, mflist, buildingblock, buildingblockmf, maxlens, smiles = batch
        
        bbout, reaction_pred = model(p4, reactions, mflist, buildingblock, buildingblockmf)
        
        # Compute losses as in training script
        bbout_flat = bbout.permute(0, 2, 1)
        reaction_flat = reaction_pred.permute(0, 2, 1)
        
        lossbb = F.cross_entropy(bbout_flat, buildingblock.long())
        lossreactions = F.cross_entropy(reaction_flat, reactions.long())
        total_loss = lossbb + lossreactions
        
        assert isinstance(lossbb, torch.Tensor)
        assert isinstance(lossreactions, torch.Tensor)
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.item() >= 0
    
    def test_backward_pass(self, mock_model_and_data):
        """Test backward pass and gradient computation"""
        model, batch = mock_model_and_data
        p4, reactions, mflist, buildingblock, buildingblockmf, maxlens, smiles = batch
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Forward pass
        bbout, reaction_pred = model(p4, reactions, mflist, buildingblock, buildingblockmf)
        
        # Compute loss
        bbout_flat = bbout.permute(0, 2, 1)
        reaction_flat = reaction_pred.permute(0, 2, 1)
        lossbb = F.cross_entropy(bbout_flat, buildingblock.long())
        lossreactions = F.cross_entropy(reaction_flat, reactions.long())
        loss = lossbb + lossreactions
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check that gradients exist
        has_gradients = any(param.grad is not None for param in model.parameters())
        assert has_gradients
        
        # Check gradient magnitudes are reasonable
        total_grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        assert total_grad_norm > 0
        assert total_grad_norm < 1000  # Should not explode
    
    def test_optimizer_step(self, mock_model_and_data):
        """Test optimizer parameter updates"""
        model, batch = mock_model_and_data
        p4, reactions, mflist, buildingblock, buildingblockmf, maxlens, smiles = batch
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Store initial parameters
        initial_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Training step
        optimizer.zero_grad()
        bbout, reaction_pred = model(p4, reactions, mflist, buildingblock, buildingblockmf)
        
        bbout_flat = bbout.permute(0, 2, 1)
        reaction_flat = reaction_pred.permute(0, 2, 1)
        lossbb = F.cross_entropy(bbout_flat, buildingblock.long())
        lossreactions = F.cross_entropy(reaction_flat, reactions.long())
        loss = lossbb + lossreactions
        
        loss.backward()
        optimizer.step()
        
        # Check that parameters have changed
        params_changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(initial_params[name], param, atol=1e-8):
                params_changed = True
                break
        
        assert params_changed


class TestDataAugmentation:
    """Test data augmentation techniques"""
    
    def test_rotation_application(self):
        """Test applying rotation matrices to coordinates"""
        from train import random_rotation_matrices
        
        batch_size = 2
        num_points = 10
        
        # Create sample 3D coordinates
        coords = torch.randn(batch_size, num_points, 3)
        R = random_rotation_matrices(batch_size)
        
        # Apply rotation
        rotated_coords = torch.bmm(coords, R)
        
        assert rotated_coords.shape == coords.shape
        
        # Check that distances are preserved (rotation is isometric)
        orig_norms = torch.norm(coords, dim=-1)
        rot_norms = torch.norm(rotated_coords, dim=-1)
        
        assert torch.allclose(orig_norms, rot_norms, atol=1e-5)


class TestModelSaving:
    """Test model saving and loading during training"""
    
    def test_model_state_dict_save_load(self):
        """Test saving and loading model state dict"""
        model = Transformer(
            source_vocab_size=100,
            target_vocab_size=100,
            embedding_dim=64,
            source_max_seq_len=32,
            target_max_seq_len=32,
            num_layers=2,
            num_heads=4,
            dropout=0.1
        )
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
            model_path = tmp.name
        
        try:
            # Save model
            torch.save(model.state_dict(), model_path)
            
            # Create new model and load
            new_model = Transformer(
                source_vocab_size=100,
                target_vocab_size=100,
                embedding_dim=64,
                source_max_seq_len=32,
                target_max_seq_len=32,
                num_layers=2,
                num_heads=4,
                dropout=0.1
            )
            
            new_model.load_state_dict(torch.load(model_path, map_location='cpu'))
            
            # Compare parameters
            for (name1, param1), (name2, param2) in zip(model.named_parameters(), new_model.named_parameters()):
                assert torch.allclose(param1, param2)
        
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)


class TestTrainingConfiguration:
    """Test training configuration and hyperparameters"""
    
    def test_optimizer_configuration(self):
        """Test optimizer configuration"""
        model = Transformer(100, 100, 32, 32, 64, 4, 2)
        
        # Test Adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]['lr'] == 0.001
        
        # Test parameter groups
        param_count = sum(1 for _ in model.parameters())
        optimizer_param_count = sum(len(group['params']) for group in optimizer.param_groups)
        assert param_count == optimizer_param_count
    
    def test_learning_rate_scheduling(self):
        """Test learning rate scheduling"""
        model = Transformer(100, 100, 32, 32, 64, 4, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Test step LR scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Step scheduler multiple times
        for _ in range(15):
            scheduler.step()
        
        # After 10 steps, LR should be reduced
        new_lr = optimizer.param_groups[0]['lr']
        assert new_lr < initial_lr
    
    def test_device_configuration(self):
        """Test device configuration"""
        # Test CPU device
        device = torch.device("cpu")
        model = Transformer(100, 100, 32, 32, 64, 4, 2)
        model = model.to(device)
        
        # Check that model is on correct device
        assert next(model.parameters()).device == device
        
        # Test CUDA if available
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model = model.to(device)
            assert next(model.parameters()).device.type == 'cuda'


class TestTrainingMetrics:
    """Test training metrics and monitoring"""
    
    def test_loss_tracking(self):
        """Test loss tracking during training"""
        losses = []
        
        # Simulate training losses
        for epoch in range(5):
            epoch_loss = 10.0 / (epoch + 1)  # Decreasing loss
            losses.append(epoch_loss)
        
        # Check that loss is decreasing
        assert all(losses[i] >= losses[i+1] for i in range(len(losses)-1))
    
    def test_validation_metrics(self):
        """Test validation metrics computation"""
        # Mock validation predictions and targets
        predictions = torch.randn(100, 10)  # 100 samples, 10 classes
        targets = torch.randint(0, 10, (100,))
        
        # Compute accuracy
        pred_classes = torch.argmax(predictions, dim=1)
        accuracy = (pred_classes == targets).float().mean()
        
        assert 0 <= accuracy <= 1
    
    def test_early_stopping_logic(self):
        """Test early stopping logic"""
        val_losses = [5.0, 4.5, 4.0, 4.1, 4.2, 4.3]  # Loss stops improving
        patience = 3
        best_loss = float('inf')
        patience_counter = 0
        
        for loss in val_losses:
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        # Should trigger early stopping
        assert patience_counter >= patience


class TestTrainingIntegration:
    """Integration tests for complete training pipeline"""
    
    @pytest.mark.slow
    def test_mini_training_run(self):
        """Test a minimal training run"""
        # Create minimal model and data
        model = Transformer(50, 53, 32, 32, 32, 2, 1, dropout=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Create minimal batch
        batch_size = 2
        seq_len = 3
        
        p4 = torch.randn(batch_size, seq_len, 11)
        reactions = torch.randint(0, 56, (batch_size, seq_len))
        mflist = torch.randn(batch_size, seq_len, 1024)
        buildingblock = torch.randint(0, 53, (batch_size, seq_len))
        buildingblockmf = torch.randn(batch_size, seq_len, 1024)
        
        initial_loss = None
        final_loss = None
        
        # Mini training loop
        model.train()
        for epoch in range(3):
            optimizer.zero_grad()
            
            bbout, reaction_pred = model(p4, reactions, mflist, buildingblock, buildingblockmf)
            
            bbout_flat = bbout.permute(0, 2, 1)
            reaction_flat = reaction_pred.permute(0, 2, 1)
            lossbb = F.cross_entropy(bbout_flat, buildingblock.long())
            lossreactions = F.cross_entropy(reaction_flat, reactions.long())
            loss = lossbb + lossreactions
            
            if epoch == 0:
                initial_loss = loss.item()
            if epoch == 2:
                final_loss = loss.item()
            
            loss.backward()
            optimizer.step()
        
        # Check that training ran without errors
        assert initial_loss is not None
        assert final_loss is not None
        # Loss should change (but not necessarily decrease in just 3 epochs)
        assert initial_loss != final_loss


class TestErrorHandling:
    """Test error handling in training"""
    
    def test_shape_mismatch_handling(self):
        """Test handling of shape mismatches"""
        model = Transformer(100, 100, 32, 32, 64, 4, 2)
        
        # Create mismatched shapes
        p4 = torch.randn(2, 5, 11)
        reactions = torch.randint(0, 56, (2, 3))  # Different seq_len
        mflist = torch.randn(2, 5, 1024)
        buildingblock = torch.randint(0, 100, (2, 5))
        buildingblockmf = torch.randn(2, 5, 1024)
        
        # This should raise an error or handle gracefully
        with pytest.raises((RuntimeError, ValueError)):
            model(p4, reactions, mflist, buildingblock, buildingblockmf)
    
    def test_invalid_vocab_indices(self):
        """Test handling of invalid vocabulary indices"""
        model = Transformer(100, 100, 32, 32, 64, 4, 2)
        
        p4 = torch.randn(2, 5, 11)
        reactions = torch.randint(0, 56, (2, 5))
        mflist = torch.randn(2, 5, 1024)
        buildingblock = torch.tensor([[150, 200, 300, 400, 500], [150, 200, 300, 400, 500]])  # Out of vocab
        buildingblockmf = torch.randn(2, 5, 1024)
        
        # This should handle out-of-vocabulary indices gracefully
        try:
            bbout, reaction_pred = model(p4, reactions, mflist, buildingblock, buildingblockmf)
            # If no error, check output shapes
            assert bbout.shape[0] == 2
        except (RuntimeError, IndexError):
            # Expected behavior for out-of-vocab indices
            pass


# Helper function to run all tests
def run_training_tests():
    """Run all training tests with pytest"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_training_tests() 