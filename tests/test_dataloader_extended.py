import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from unittest.mock import Mock, patch
import tempfile
import pickle

# Import the modules to test
from Dataloader import Datasetp4, custom_collate_fn


class TestDatasetp4:
    """Test the Datasetp4 dataset class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        # Structure: [reactions_data, smiles_data, fingerprint_data]
        data = [
            [  # reactions data: [reaction_id, bb1_id, bb2_id, length]
                [1, 100, 101, 2],
                [2, 101, 102, 3],
                [3, 102, 103, 1],
                [4, 103, 104, 2],
                [5, 104, 105, 3],
            ],
            [  # smiles data
                "CCO",
                "CCC", 
                "CCCO",
                "CCCCO",
                "CCCCCO"
            ],
            [  # intermediate molecular fingerprints
                np.random.rand(1024).astype(np.float32),
                np.random.rand(1024).astype(np.float32),
                np.random.rand(1024).astype(np.float32),
                np.random.rand(1024).astype(np.float32),
                np.random.rand(1024).astype(np.float32),
            ]
        ]
        
        # p4 data (pharmacophore features)
        p4 = [np.random.rand(50, 11).astype(np.float32) for _ in range(5)]
        
        # Building block fingerprints
        fingerprint_list = [np.random.rand(1024).astype(np.float32) for _ in range(200)]
        
        # Indexes to use
        indexes = [0, 1, 2, 3, 4]
        
        return data, p4, fingerprint_list, indexes
    
    def test_dataset_initialization(self, sample_data):
        """Test dataset initialization"""
        data, p4, fingerprint_list, indexes = sample_data
        
        dataset = Datasetp4(data, p4, indexes, fingerprint_list)
        
        assert len(dataset) == len(indexes)
        assert dataset.diclen == len(fingerprint_list)
        assert len(dataset.data) == len(data[0])
        assert len(dataset.smiles) == len(data[1])
        assert len(dataset.imf) == len(data[2])
    
    def test_dataset_length(self, sample_data):
        """Test dataset length property"""
        data, p4, fingerprint_list, indexes = sample_data
        
        # Test with different index lengths
        for idx_len in [1, 3, 5]:
            test_indexes = indexes[:idx_len]
            dataset = Datasetp4(data, p4, test_indexes, fingerprint_list)
            assert len(dataset) == idx_len
    
    def test_dataset_getitem_basic(self, sample_data):
        """Test basic item retrieval"""
        data, p4, fingerprint_list, indexes = sample_data
        
        dataset = Datasetp4(data, p4, indexes, fingerprint_list)
        
        # Get first item (note: dataset uses idx-1 internally)
        item = dataset[1]
        
        # Check return structure: (p4T, reactionsT, mflistT, buildingblockT, buildingblockmfT, maxlens, smiles)
        assert len(item) == 7
        
        # Check tensor types
        assert isinstance(item[0], torch.Tensor)  # p4T
        assert isinstance(item[1], torch.Tensor)  # reactionsT
        assert isinstance(item[2], torch.Tensor)  # mflistT
        assert isinstance(item[3], torch.Tensor)  # buildingblockT
        assert isinstance(item[4], torch.Tensor)  # buildingblockmfT
        assert isinstance(item[5], list)          # maxlens
        assert isinstance(item[6], str)           # smiles
    
    def test_dataset_getitem_shapes(self, sample_data):
        """Test shapes of returned tensors"""
        data, p4, fingerprint_list, indexes = sample_data
        
        dataset = Datasetp4(data, p4, indexes, fingerprint_list)
        item = dataset[1]
        
        p4T, reactionsT, mflistT, buildingblockT, buildingblockmfT, maxlens, smiles = item
        
        # p4 should maintain original shape
        assert p4T.shape == torch.Size([50, 11])
        
        # Other tensors should have consistent sequence length
        seq_len = reactionsT.shape[0]
        assert mflistT.shape == torch.Size([seq_len, 1024])
        assert buildingblockT.shape == torch.Size([seq_len])
        assert buildingblockmfT.shape == torch.Size([seq_len, 1024])
    
    def test_dataset_getitem_sequence_construction(self, sample_data):
        """Test that sequences are constructed correctly"""
        data, p4, fingerprint_list, indexes = sample_data
        
        dataset = Datasetp4(data, p4, indexes, fingerprint_list)
        item = dataset[1]
        
        p4T, reactionsT, mflistT, buildingblockT, buildingblockmfT, maxlens, smiles = item
        
        # Check that sequences start with appropriate tokens
        assert reactionsT[0].item() == 0  # Should start with 0 token
        
        # Check that building block sequence ends with special token
        assert buildingblockT[-1].item() == dataset.diclen + 1
        
        # Check that molecular fingerprint sequence includes start token
        start_fp = mflistT[-1]  # Last element should be start token
        assert start_fp[0].item() == 1.0  # First bit should be 1
        assert torch.all(start_fp[1:] == 0).item()  # Rest should be 0
    
    def test_dataset_edge_cases(self, sample_data):
        """Test edge cases and boundary conditions"""
        data, p4, fingerprint_list, indexes = sample_data
        
        dataset = Datasetp4(data, p4, indexes, fingerprint_list)
        
        # Test first valid index (should be 1, not 0 due to idx-1)
        item = dataset[1]
        assert item is not None
        
        # Test last valid index
        item = dataset[len(indexes)]
        assert item is not None
    
    def test_dataset_data_consistency(self, sample_data):
        """Test data consistency across multiple retrievals"""
        data, p4, fingerprint_list, indexes = sample_data
        
        dataset = Datasetp4(data, p4, indexes, fingerprint_list)
        
        # Get same item multiple times
        item1 = dataset[1]
        item2 = dataset[1]
        
        # Should be identical
        assert torch.allclose(item1[0], item2[0])  # p4
        assert torch.allclose(item1[1], item2[1])  # reactions
        assert torch.allclose(item1[2], item2[2])  # mflist
        assert torch.allclose(item1[3], item2[3])  # buildingblock
        assert torch.allclose(item1[4], item2[4])  # buildingblockmf
    
    def test_dataset_different_lengths(self, sample_data):
        """Test dataset with different sequence lengths"""
        data, p4, fingerprint_list, indexes = sample_data
        
        # Modify data to have different lengths
        data[0][0][3] = 1  # length = 1
        data[0][1][3] = 3  # length = 3
        
        dataset = Datasetp4(data, p4, indexes, fingerprint_list)
        
        item1 = dataset[1]  # length = 1
        item2 = dataset[2]  # length = 3
        
        # Should have different sequence lengths
        assert item1[1].shape[0] != item2[1].shape[0]


class TestCustomCollateFn:
    """Test the custom_collate_fn function"""
    
    @pytest.fixture
    def sample_batch(self):
        """Create sample batch for testing"""
        batch_size = 3
        seq_lens = [5, 7, 4]  # Different sequence lengths
        
        batch = []
        for i, seq_len in enumerate(seq_lens):
            p4 = torch.randn(50, 11)
            reactions = torch.randint(0, 56, (seq_len,))
            mflist = torch.randn(seq_len, 1024)
            buildingblock = torch.randint(0, 100, (seq_len,))
            buildingblockmf = torch.randn(seq_len, 1024)
            maxlens = [seq_len]
            smiles = f"smiles_{i}"
            
            batch.append((p4, reactions, mflist, buildingblock, buildingblockmf, maxlens, smiles))
        
        return batch
    
    def test_collate_fn_basic(self, sample_batch):
        """Test basic functionality of collate function"""
        result = custom_collate_fn(sample_batch)
        
        # Should return 7 elements
        assert len(result) == 7
        
        # Check return types
        p4_padded, reactions_padded, mflist_padded, buildingblock_padded, buildingblockmf_padded, maxlens, smiles = result
        
        assert isinstance(p4_padded, torch.Tensor)
        assert isinstance(reactions_padded, torch.Tensor)
        assert isinstance(mflist_padded, torch.Tensor)
        assert isinstance(buildingblock_padded, torch.Tensor)
        assert isinstance(buildingblockmf_padded, torch.Tensor)
        assert isinstance(maxlens, tuple)
        assert isinstance(smiles, tuple)
    
    def test_collate_fn_padding(self, sample_batch):
        """Test that sequences are properly padded"""
        result = custom_collate_fn(sample_batch)
        p4_padded, reactions_padded, mflist_padded, buildingblock_padded, buildingblockmf_padded, maxlens, smiles = result
        
        batch_size = len(sample_batch)
        max_seq_len = max(sample_batch[i][1].shape[0] for i in range(batch_size))
        
        # Check padded shapes
        assert p4_padded.shape == torch.Size([batch_size, 50, 11])
        assert reactions_padded.shape == torch.Size([batch_size, max_seq_len])
        assert mflist_padded.shape == torch.Size([batch_size, max_seq_len, 1024])
        assert buildingblock_padded.shape == torch.Size([batch_size, max_seq_len])
        assert buildingblockmf_padded.shape == torch.Size([batch_size, max_seq_len, 1024])
    
    def test_collate_fn_padding_values(self, sample_batch):
        """Test that padding values are correct (should be 0)"""
        result = custom_collate_fn(sample_batch)
        p4_padded, reactions_padded, mflist_padded, buildingblock_padded, buildingblockmf_padded, maxlens, smiles = result
        
        # Check that padding is with zeros
        # For the shortest sequence, check that padded positions are zero
        shortest_idx = 2  # seq_len = 4
        shortest_len = sample_batch[shortest_idx][1].shape[0]
        
        # Check reactions padding
        padded_reactions = reactions_padded[shortest_idx, shortest_len:]
        assert torch.all(padded_reactions == 0)
        
        # Check building block padding  
        padded_bb = buildingblock_padded[shortest_idx, shortest_len:]
        assert torch.all(padded_bb == 0)
    
    def test_collate_fn_batch_first(self, sample_batch):
        """Test that batch dimension is first"""
        result = custom_collate_fn(sample_batch)
        p4_padded, reactions_padded, mflist_padded, buildingblock_padded, buildingblockmf_padded, maxlens, smiles = result
        
        batch_size = len(sample_batch)
        
        # All tensors should have batch as first dimension
        assert p4_padded.shape[0] == batch_size
        assert reactions_padded.shape[0] == batch_size
        assert mflist_padded.shape[0] == batch_size
        assert buildingblock_padded.shape[0] == batch_size
        assert buildingblockmf_padded.shape[0] == batch_size
    
    def test_collate_fn_preserve_data(self, sample_batch):
        """Test that original data is preserved (not just padding added)"""
        result = custom_collate_fn(sample_batch)
        p4_padded, reactions_padded, mflist_padded, buildingblock_padded, buildingblockmf_padded, maxlens, smiles = result
        
        # Check that original data is preserved
        for i, (p4, reactions, mflist, buildingblock, buildingblockmf, orig_maxlens, orig_smiles) in enumerate(sample_batch):
            seq_len = reactions.shape[0]
            
            # Check p4 data
            assert torch.allclose(p4_padded[i], p4)
            
            # Check sequence data (non-padded parts)
            assert torch.allclose(reactions_padded[i, :seq_len], reactions)
            assert torch.allclose(mflist_padded[i, :seq_len], mflist)
            assert torch.allclose(buildingblock_padded[i, :seq_len], buildingblock)
            assert torch.allclose(buildingblockmf_padded[i, :seq_len], buildingblockmf)
            
            # Check metadata
            assert smiles[i] == orig_smiles


class TestDataLoaderIntegration:
    """Test DataLoader integration with Datasetp4 and custom_collate_fn"""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for integration testing"""
        # Create more extensive sample data
        data = [
            [[i, 100+i, 101+i, (i % 3) + 1] for i in range(10)],  # reactions
            [f"smiles_{i}" for i in range(10)],                    # smiles
            [np.random.rand(1024).astype(np.float32) for i in range(10)]  # fingerprints
        ]
        
        p4 = [np.random.rand(50, 11).astype(np.float32) for _ in range(10)]
        fingerprint_list = [np.random.rand(1024).astype(np.float32) for _ in range(200)]
        indexes = list(range(10))
        
        return Datasetp4(data, p4, indexes, fingerprint_list)
    
    def test_dataloader_basic_functionality(self, sample_dataset):
        """Test basic DataLoader functionality"""
        batch_size = 3
        dataloader = DataLoader(
            sample_dataset, 
            batch_size=batch_size, 
            collate_fn=custom_collate_fn,
            shuffle=False
        )
        
        # Get first batch
        batch = next(iter(dataloader))
        
        assert len(batch) == 7
        assert batch[0].shape[0] == batch_size  # p4
        assert batch[1].shape[0] == batch_size  # reactions
    
    def test_dataloader_iteration(self, sample_dataset):
        """Test iterating through DataLoader"""
        batch_size = 4
        dataloader = DataLoader(
            sample_dataset,
            batch_size=batch_size,
            collate_fn=custom_collate_fn,
            shuffle=False
        )
        
        batches = list(dataloader)
        
        # Should have 3 batches: [4, 4, 2] samples
        assert len(batches) == 3
        assert batches[0][0].shape[0] == 4  # First batch
        assert batches[1][0].shape[0] == 4  # Second batch
        assert batches[2][0].shape[0] == 2  # Last batch (remainder)
    
    def test_dataloader_shuffle(self, sample_dataset):
        """Test DataLoader shuffling"""
        batch_size = 2
        
        # Create two dataloaders with same seed
        dataloader1 = DataLoader(
            sample_dataset,
            batch_size=batch_size,
            collate_fn=custom_collate_fn,
            shuffle=True
        )
        
        dataloader2 = DataLoader(
            sample_dataset,
            batch_size=batch_size,
            collate_fn=custom_collate_fn,
            shuffle=True
        )
        
        # Get first batches
        batch1 = next(iter(dataloader1))
        batch2 = next(iter(dataloader2))
        
        # They should potentially be different due to shuffling
        # (though they might be the same by chance)
        assert batch1[0].shape == batch2[0].shape  # Same shape
    
    def test_dataloader_drop_last(self, sample_dataset):
        """Test DataLoader drop_last functionality"""
        batch_size = 3
        
        # Without drop_last
        dataloader1 = DataLoader(
            sample_dataset,
            batch_size=batch_size,
            collate_fn=custom_collate_fn,
            drop_last=False
        )
        
        # With drop_last
        dataloader2 = DataLoader(
            sample_dataset,
            batch_size=batch_size,
            collate_fn=custom_collate_fn,
            drop_last=True
        )
        
        batches1 = list(dataloader1)
        batches2 = list(dataloader2)
        
        # drop_last should result in fewer batches
        assert len(batches2) <= len(batches1)


class TestDataValidation:
    """Test data validation and error handling"""
    
    def test_invalid_index_handling(self):
        """Test handling of invalid indices"""
        data = [
            [[1, 100, 101, 2]],
            ["CCO"],
            [np.random.rand(1024).astype(np.float32)]
        ]
        p4 = [np.random.rand(50, 11).astype(np.float32)]
        fingerprint_list = [np.random.rand(1024).astype(np.float32) for _ in range(10)]
        indexes = [0]
        
        dataset = Datasetp4(data, p4, indexes, fingerprint_list)
        
        # Test accessing invalid index
        with pytest.raises(IndexError):
            dataset[10]  # Out of range
    
    def test_empty_dataset(self):
        """Test behavior with empty dataset"""
        data = [[], [], []]
        p4 = []
        fingerprint_list = []
        indexes = []
        
        dataset = Datasetp4(data, p4, indexes, fingerprint_list)
        
        assert len(dataset) == 0
    
    def test_data_type_consistency(self):
        """Test that data types are consistent"""
        data = [
            [[1, 100, 101, 2]],
            ["CCO"],
            [np.random.rand(1024).astype(np.float32)]
        ]
        p4 = [np.random.rand(50, 11).astype(np.float32)]
        fingerprint_list = [np.random.rand(1024).astype(np.float32) for _ in range(200)]
        indexes = [0]
        
        dataset = Datasetp4(data, p4, indexes, fingerprint_list)
        item = dataset[1]
        
        # Check tensor dtypes
        assert item[0].dtype == torch.float32  # p4
        assert item[1].dtype == torch.float32  # reactions
        assert item[2].dtype == torch.float32  # mflist
        assert item[3].dtype == torch.float32  # buildingblock
        assert item[4].dtype == torch.float32  # buildingblockmf


class TestMemoryEfficiency:
    """Test memory efficiency and performance"""
    
    def test_memory_usage_large_dataset(self):
        """Test memory usage with larger dataset"""
        # Create larger dataset
        num_samples = 100
        data = [
            [[i, 100+i, 101+i, (i % 3) + 1] for i in range(num_samples)],
            [f"smiles_{i}" for i in range(num_samples)],
            [np.random.rand(1024).astype(np.float32) for i in range(num_samples)]
        ]
        
        p4 = [np.random.rand(50, 11).astype(np.float32) for _ in range(num_samples)]
        fingerprint_list = [np.random.rand(1024).astype(np.float32) for _ in range(1000)]
        indexes = list(range(num_samples))
        
        dataset = Datasetp4(data, p4, indexes, fingerprint_list)
        
        # Test that dataset can be created and accessed
        assert len(dataset) == num_samples
        
        # Test accessing random samples
        import random
        for _ in range(10):
            idx = random.randint(1, num_samples)
            item = dataset[idx]
            assert item is not None
    
    def test_batch_processing_performance(self):
        """Test batch processing performance"""
        # Create dataset
        num_samples = 50
        data = [
            [[i, 100+i, 101+i, (i % 3) + 1] for i in range(num_samples)],
            [f"smiles_{i}" for i in range(num_samples)],
            [np.random.rand(1024).astype(np.float32) for i in range(num_samples)]
        ]
        
        p4 = [np.random.rand(50, 11).astype(np.float32) for _ in range(num_samples)]
        fingerprint_list = [np.random.rand(1024).astype(np.float32) for _ in range(200)]
        indexes = list(range(num_samples))
        
        dataset = Datasetp4(data, p4, indexes, fingerprint_list)
        
        # Test different batch sizes
        for batch_size in [1, 4, 8, 16]:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=custom_collate_fn,
                shuffle=False
            )
            
            # Process all batches
            batch_count = 0
            for batch in dataloader:
                assert len(batch) == 7
                batch_count += 1
            
            expected_batches = (num_samples + batch_size - 1) // batch_size
            assert batch_count == expected_batches


# Helper function to run all tests
def run_dataloader_tests():
    """Run all dataloader tests with pytest"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_dataloader_tests() 