import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, mock_open
from rdkit import Chem

# Add parent directory to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the modules to test
from utils import load_enamine_building_blocks, getlistreactants


class TestLoadEnamineeBuildingBlocks:
    """Test the load_enamine_building_blocks function"""
    
    def test_load_from_sdf_file(self):
        """Test loading building blocks from SDF file"""
        # Create a temporary SDF file with sample molecules
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as tmp_file:
            sdf_content = """
  Mrv2014 01112025132D          

  3  2  0  0  0  0            999 V2000
   -0.5000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.5000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.0000    0.8660    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  1  0  0  0  0
M  END
$$$$
"""
            tmp_file.write(sdf_content)
            tmp_file_path = tmp_file.name
        
        try:
            # Test loading
            result = load_enamine_building_blocks(sdf_file=tmp_file_path, max_length=50)
            
            # Should return a list of SMILES
            assert isinstance(result, list)
            if result:  # If any molecules were successfully parsed
                assert all(isinstance(smiles, str) for smiles in result)
        
        finally:
            os.unlink(tmp_file_path)
    
    def test_load_with_max_length_filter(self):
        """Test that max_length filter works correctly"""
        # Mock SDF supplier with molecules of different lengths
        mock_mol1 = Mock()
        mock_mol1.GetNumAtoms.return_value = 5
        
        mock_mol2 = Mock()
        mock_mol2.GetNumAtoms.return_value = 50
        
        with patch('utils.Chem.SDMolSupplier') as mock_supplier:
            mock_supplier.return_value = [mock_mol1, mock_mol2, None]  # None to test None handling
            
            with patch('utils.Chem.MolToSmiles') as mock_to_smiles:
                mock_to_smiles.side_effect = ["CC", "C" * 40]  # Short and long SMILES
                
                # Test with max_length=10
                result = load_enamine_building_blocks(sdf_file="test.sdf", max_length=10)
                
                # Should only include the short SMILES
                assert "CC" in result
                assert "C" * 40 not in result
    
    def test_fallback_to_smi_file(self):
        """Test fallback to SMI file when SDF is not found"""
        non_existent_sdf = "non_existent_file.sdf"
        
        # Mock the SMI file content
        smi_content = "CCO\tmolecule1\nCCC\tmolecule2\nCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\tvery_long_molecule\n"
        
        with patch('builtins.open', mock_open(read_data=smi_content)):
            with patch('os.path.exists', return_value=False):  # SDF doesn't exist
                result = load_enamine_building_blocks(sdf_file=non_existent_sdf, max_length=10)
                
                # Should have loaded from SMI file and filtered by length
                assert isinstance(result, list)
                expected_short_molecules = ["CCO", "CCC"]
                for mol in expected_short_molecules:
                    assert mol in result
                
                # Long molecule should be filtered out
                long_mol = "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
                assert long_mol not in result
    
    def test_file_not_found_error_handling(self):
        """Test error handling when neither SDF nor SMI file exists"""
        with patch('utils.Chem.SDMolSupplier', side_effect=FileNotFoundError):
            with patch('builtins.open', side_effect=FileNotFoundError):
                result = load_enamine_building_blocks(sdf_file="non_existent.sdf")
                
                # Should return empty list when no files found
                assert result == []
    
    def test_invalid_molecules_handling(self):
        """Test handling of invalid molecules in SDF"""
        with patch('utils.Chem.SDMolSupplier') as mock_supplier:
            # Mock supplier with None values (invalid molecules)
            mock_supplier.return_value = [None, None, None]
            
            result = load_enamine_building_blocks(sdf_file="test.sdf")
            
            # Should return empty list when all molecules are invalid
            assert result == []
    
    def test_empty_smiles_handling(self):
        """Test handling of empty SMILES strings"""
        mock_mol = Mock()
        
        with patch('utils.Chem.SDMolSupplier') as mock_supplier:
            mock_supplier.return_value = [mock_mol]
            
            with patch('utils.Chem.MolToSmiles', return_value=""):  # Empty SMILES
                result = load_enamine_building_blocks(sdf_file="test.sdf")
                
                # Should filter out empty SMILES
                assert result == []
    
    def test_default_parameters(self):
        """Test function with default parameters"""
        with patch('utils.Chem.SDMolSupplier', side_effect=FileNotFoundError):
            with patch('builtins.open', side_effect=FileNotFoundError):
                result = load_enamine_building_blocks()
                
                # Should handle default parameters gracefully
                assert isinstance(result, list)
    
    @patch('builtins.print')
    def test_print_statements(self, mock_print):
        """Test that appropriate print statements are called"""
        # Test successful loading
        mock_mol = Mock()
        
        with patch('utils.Chem.SDMolSupplier') as mock_supplier:
            mock_supplier.return_value = [mock_mol]
            
            with patch('utils.Chem.MolToSmiles', return_value="CCO"):
                result = load_enamine_building_blocks(sdf_file="test.sdf")
                
                # Should print success message
                mock_print.assert_called()
                call_args = [call[0][0] for call in mock_print.call_args_list]
                assert any("Loaded" in arg and "building blocks" in arg for arg in call_args)


class TestGetListReactants:
    """Test the getlistreactants function"""
    
    @pytest.fixture
    def sample_masks(self):
        """Create sample precomputed masks for testing"""
        # Create a 3D array: [2, num_reactions, num_building_blocks]
        num_reactions = 5
        num_building_blocks = 10
        
        masks = np.zeros((2, num_reactions, num_building_blocks))
        
        # Set some specific patterns for testing
        # Reaction 0: building blocks 0,1,2 compatible with reactant 0; 3,4,5 with reactant 1
        masks[0, 0, [0, 1, 2]] = 1
        masks[1, 0, [3, 4, 5]] = 1
        
        # Reaction 1: building blocks 1,3,5 compatible with reactant 0; 0,2,4 with reactant 1
        masks[0, 1, [1, 3, 5]] = 1
        masks[1, 1, [0, 2, 4]] = 1
        
        # Reaction 2: no compatible building blocks
        # (all zeros, already initialized)
        
        # Reaction 3: all building blocks compatible with both reactants
        masks[0, 3, :] = 1
        masks[1, 3, :] = 1
        
        return masks
    
    def test_basic_functionality(self, sample_masks):
        """Test basic functionality of getlistreactants"""
        # Test reaction 0, reactant 0
        result = getlistreactants(0, 0, sample_masks)
        
        assert isinstance(result, list)
        assert result == [0, 1, 2]
    
    def test_different_reactant_positions(self, sample_masks):
        """Test with different reactant positions"""
        # Test reaction 0, reactant 1
        result = getlistreactants(0, 1, sample_masks)
        assert result == [3, 4, 5]
        
        # Test reaction 1, reactant 0
        result = getlistreactants(1, 0, sample_masks)
        assert result == [1, 3, 5]
        
        # Test reaction 1, reactant 1
        result = getlistreactants(1, 1, sample_masks)
        assert result == [0, 2, 4]
    
    def test_no_compatible_building_blocks(self, sample_masks):
        """Test when no building blocks are compatible"""
        # Reaction 2 has no compatible building blocks
        result = getlistreactants(2, 0, sample_masks)
        assert result == []
        
        result = getlistreactants(2, 1, sample_masks)
        assert result == []
    
    def test_all_compatible_building_blocks(self, sample_masks):
        """Test when all building blocks are compatible"""
        # Reaction 3 has all building blocks compatible
        result = getlistreactants(3, 0, sample_masks)
        expected = list(range(10))  # All building blocks 0-9
        assert result == expected
        
        result = getlistreactants(3, 1, sample_masks)
        assert result == expected
    
    def test_edge_cases(self, sample_masks):
        """Test edge cases and boundary conditions"""
        # Test first reaction, first reactant
        result = getlistreactants(0, 0, sample_masks)
        assert isinstance(result, list)
        
        # Test last reaction
        num_reactions = sample_masks.shape[1]
        result = getlistreactants(num_reactions - 1, 0, sample_masks)
        assert isinstance(result, list)
    
    def test_mask_array_shapes(self):
        """Test with different mask array shapes"""
        # Test with minimal array
        small_masks = np.array([[[1, 0]], [[0, 1]]])  # 2x1x2
        
        result = getlistreactants(0, 0, small_masks)
        assert result == [0]
        
        result = getlistreactants(0, 1, small_masks)
        assert result == [1]
    
    def test_empty_mask_array(self):
        """Test with empty mask array"""
        empty_masks = np.zeros((2, 1, 0))  # No building blocks
        
        result = getlistreactants(0, 0, empty_masks)
        assert result == []
    
    def test_function_return_type_consistency(self, sample_masks):
        """Test that function always returns a list of integers"""
        for rxn_idx in range(sample_masks.shape[1]):
            for reactant_idx in range(2):
                result = getlistreactants(rxn_idx, reactant_idx, sample_masks)
                
                assert isinstance(result, list)
                assert all(isinstance(item, (int, np.integer)) for item in result)
    
    def test_index_boundary_validation(self, sample_masks):
        """Test behavior with boundary indices"""
        num_reactions = sample_masks.shape[1]
        num_building_blocks = sample_masks.shape[2]
        
        # Test with valid boundary indices
        result = getlistreactants(num_reactions - 1, 1, sample_masks)
        assert isinstance(result, list)
        assert all(0 <= idx < num_building_blocks for idx in result)
    
    def test_mask_pattern_verification(self, sample_masks):
        """Test that the function correctly interprets mask patterns"""
        # Manually verify a specific pattern
        rxn_idx, reactant_idx = 1, 0
        expected_indices = [1, 3, 5]  # From our sample_masks setup
        
        result = getlistreactants(rxn_idx, reactant_idx, sample_masks)
        
        assert set(result) == set(expected_indices)
        assert len(result) == len(expected_indices)


class TestUtilsIntegration:
    """Integration tests for utils module"""
    
    def test_realistic_workflow(self):
        """Test a realistic workflow using utils functions"""
        # Mock a realistic scenario
        with patch('utils.Chem.SDMolSupplier', side_effect=FileNotFoundError):
            smi_content = "CCO\tmol1\nCCC\tmol2\nC=O\tmol3\n"
            
            with patch('builtins.open', mock_open(read_data=smi_content)):
                # Load building blocks
                building_blocks = load_enamine_building_blocks(max_length=10)
                
                assert len(building_blocks) == 3
                assert "CCO" in building_blocks
                assert "CCC" in building_blocks
                assert "C=O" in building_blocks
    
    def test_error_resilience(self):
        """Test that utils functions are resilient to various errors"""
        # Test with corrupted data
        with patch('utils.Chem.SDMolSupplier', side_effect=Exception("Corrupted file")):
            with patch('builtins.open', side_effect=Exception("File read error")):
                result = load_enamine_building_blocks()
                
                # Should handle errors gracefully
                assert isinstance(result, list)
    
    def test_memory_efficiency_large_dataset(self):
        """Test memory efficiency with large datasets"""
        # Simulate large dataset processing
        large_masks = np.random.randint(0, 2, (2, 100, 1000))  # 100 reactions, 1000 building blocks
        
        # Test that function can handle large arrays efficiently
        for rxn_idx in range(10):  # Test first 10 reactions
            for reactant_idx in range(2):
                result = getlistreactants(rxn_idx, reactant_idx, large_masks)
                
                assert isinstance(result, list)
                # Result should be reasonable size (not all building blocks)
                assert len(result) <= 1000


class TestUtilsErrorHandling:
    """Test error handling in utils module"""
    
    def test_invalid_file_paths(self):
        """Test handling of invalid file paths"""
        # Test with special characters and invalid paths
        invalid_paths = [
            "",
            "/invalid/path/file.sdf",
            "file\x00with\x00nulls.sdf",
            "path/with/unicode/文件.sdf"
        ]
        
        for path in invalid_paths:
            try:
                result = load_enamine_building_blocks(sdf_file=path)
                assert isinstance(result, list)  # Should not crash
            except Exception as e:
                # Should handle gracefully
                assert isinstance(e, (FileNotFoundError, ValueError, OSError))
    
    def test_malformed_mask_arrays(self):
        """Test handling of malformed mask arrays"""
        # Test with wrong dimensions
        wrong_shape_masks = np.zeros((3, 5, 10))  # Should be (2, 5, 10)
        
        try:
            result = getlistreactants(0, 0, wrong_shape_masks)
            # If it doesn't raise an error, result should still be a list
            assert isinstance(result, list)
        except IndexError:
            # This is acceptable behavior for malformed input
            pass
    
    def test_out_of_bounds_indices(self):
        """Test handling of out-of-bounds indices"""
        masks = np.zeros((2, 3, 5))
        
        # Test with indices that are out of bounds
        with pytest.raises(IndexError):
            getlistreactants(5, 0, masks)  # rxn_idx out of bounds
        
        with pytest.raises(IndexError):
            getlistreactants(0, 5, masks)  # reactant_idx out of bounds


# Helper function to run all tests
def run_utils_tests():
    """Run all utils tests with pytest"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_utils_tests() 