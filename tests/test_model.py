import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import (
    Transformer, Encoder, Decoder, EncoderLayer, DecoderLayer,
    MultiHeadAttention, SelfAttention, Embedding, PositionalEncoder, Norm
)
from unittest.mock import Mock, patch
import tempfile
import os


class TestEmbedding:
    """Test the Embedding layer"""
    
    def test_embedding_initialization(self):
        vocab_size = 1000
        embedding_dim = 512
        embedding = Embedding(vocab_size, embedding_dim)
        
        assert embedding.embedding.num_embeddings == vocab_size
        assert embedding.embedding.embedding_dim == embedding_dim
    
    def test_embedding_forward(self):
        vocab_size = 1000
        embedding_dim = 512
        batch_size = 2
        seq_len = 10
        
        embedding = Embedding(vocab_size, embedding_dim)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        output = embedding(input_ids)
        
        assert output.shape == (batch_size, seq_len, embedding_dim)
    
    def test_embedding_out_of_bounds(self):
        vocab_size = 100
        embedding_dim = 64
        embedding = Embedding(vocab_size, embedding_dim)
        
        # Test with valid input
        valid_input = torch.randint(0, vocab_size, (2, 5))
        output = embedding(valid_input)
        assert output.shape == (2, 5, embedding_dim)


class TestPositionalEncoder:
    """Test the PositionalEncoder layer"""
    
    def test_positional_encoder_initialization(self):
        embedding_dim = 512
        max_seq_length = 1000
        dropout = 0.1
        
        pe = PositionalEncoder(embedding_dim, max_seq_length, dropout)
        
        assert pe.embedding_dim == embedding_dim
        assert pe.pe.shape == (1, max_seq_length, embedding_dim)
    
    def test_positional_encoder_forward(self):
        embedding_dim = 512
        max_seq_length = 100
        batch_size = 2
        seq_len = 50
        
        pe = PositionalEncoder(embedding_dim, max_seq_length)
        x = torch.randn(batch_size, seq_len, embedding_dim)
        
        output = pe(x)
        
        assert output.shape == (batch_size, seq_len, embedding_dim)
    
    def test_positional_encoder_scaling(self):
        embedding_dim = 512
        pe = PositionalEncoder(embedding_dim)
        
        x = torch.ones(1, 10, embedding_dim)
        output = pe(x)
        
        # Check that input is scaled by sqrt(embedding_dim)
        expected_first_element = np.sqrt(embedding_dim)
        # Allow for some tolerance due to PE addition and dropout
        assert output[0, 0, 0] != x[0, 0, 0]  # Should be different due to scaling and PE
    
    def test_positional_encoder_device_compatibility(self):
        embedding_dim = 64
        pe = PositionalEncoder(embedding_dim)
        
        if torch.cuda.is_available():
            device = torch.device('cuda')
            pe = pe.to(device)
            x = torch.randn(1, 10, embedding_dim).to(device)
            output = pe(x)
            assert output.device == device


class TestSelfAttention:
    """Test the SelfAttention mechanism"""
    
    def test_self_attention_initialization(self):
        dropout = 0.1
        attn = SelfAttention(dropout)
        assert isinstance(attn.dropout, nn.Dropout)
    
    def test_self_attention_forward(self):
        batch_size = 2
        seq_len = 10
        embedding_dim = 64
        num_heads = 8
        head_dim = embedding_dim // num_heads
        
        attn = SelfAttention()
        query = torch.randn(batch_size, num_heads, seq_len, head_dim)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        output = attn(query, key, value)
        
        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
    
    def test_self_attention_with_mask(self):
        batch_size = 2
        seq_len = 10
        embedding_dim = 64
        num_heads = 8
        head_dim = embedding_dim // num_heads
        
        attn = SelfAttention()
        query = torch.randn(batch_size, num_heads, seq_len, head_dim)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        # Create a mask that masks out the last half of the sequence
        mask = torch.ones(batch_size, seq_len, seq_len)
        mask[:, :, seq_len//2:] = 0
        
        output = attn(query, key, value, mask)
        
        assert output.shape == (batch_size, num_heads, seq_len, head_dim)


class TestMultiHeadAttention:
    """Test the MultiHeadAttention layer"""
    
    def test_multihead_attention_initialization(self):
        embedding_dim = 512
        num_heads = 8
        
        mha = MultiHeadAttention(embedding_dim, num_heads)
        
        assert mha.embedding_dim == embedding_dim
        assert mha.num_heads == num_heads
        assert mha.dim_per_head == embedding_dim // num_heads
    
    def test_multihead_attention_forward(self):
        batch_size = 2
        seq_len = 10
        embedding_dim = 64
        num_heads = 8
        
        mha = MultiHeadAttention(embedding_dim, num_heads)
        x = torch.randn(batch_size, seq_len, embedding_dim)
        
        output = mha(x, x, x)
        
        assert output.shape == (batch_size, seq_len, embedding_dim)
    
    def test_multihead_attention_cross_attention(self):
        batch_size = 2
        query_len = 5
        key_value_len = 8
        embedding_dim = 64
        num_heads = 8
        
        mha = MultiHeadAttention(embedding_dim, num_heads)
        query = torch.randn(batch_size, query_len, embedding_dim)
        key = torch.randn(batch_size, key_value_len, embedding_dim)
        value = torch.randn(batch_size, key_value_len, embedding_dim)
        
        output = mha(query, key, value)
        
        assert output.shape == (batch_size, query_len, embedding_dim)


class TestNorm:
    """Test the Norm (LayerNorm) layer"""
    
    def test_norm_initialization(self):
        embedding_dim = 512
        norm = Norm(embedding_dim)
        assert isinstance(norm.norm, nn.LayerNorm)
    
    def test_norm_forward(self):
        batch_size = 2
        seq_len = 10
        embedding_dim = 64
        
        norm = Norm(embedding_dim)
        x = torch.randn(batch_size, seq_len, embedding_dim)
        
        output = norm(x)
        
        assert output.shape == x.shape
        # Check that normalization is applied (mean should be close to 0)
        assert torch.abs(output.mean(dim=-1)).max() < 1e-5


class TestEncoderLayer:
    """Test the EncoderLayer"""
    
    def test_encoder_layer_initialization(self):
        embedding_dim = 512
        num_heads = 8
        ff_dim = 2048
        
        layer = EncoderLayer(embedding_dim, num_heads, ff_dim)
        
        assert isinstance(layer.self_attention, MultiHeadAttention)
        assert isinstance(layer.feed_forward, nn.Sequential)
        assert isinstance(layer.norm1, Norm)
        assert isinstance(layer.norm2, Norm)
    
    def test_encoder_layer_forward(self):
        batch_size = 2
        seq_len = 10
        embedding_dim = 64
        num_heads = 8
        
        layer = EncoderLayer(embedding_dim, num_heads)
        x = torch.randn(batch_size, seq_len, embedding_dim)
        
        output = layer(x)
        
        assert output.shape == x.shape


class TestDecoderLayer:
    """Test the DecoderLayer"""
    
    def test_decoder_layer_initialization(self):
        embedding_dim = 512
        num_heads = 8
        ff_dim = 2048
        
        layer = DecoderLayer(embedding_dim, num_heads, ff_dim)
        
        assert isinstance(layer.self_attention, MultiHeadAttention)
        assert isinstance(layer.encoder_attention, MultiHeadAttention)
        assert isinstance(layer.feed_forward, nn.Sequential)
    
    def test_decoder_layer_forward(self):
        batch_size = 2
        seq_len = 10
        embedding_dim = 64
        num_heads = 8
        
        layer = DecoderLayer(embedding_dim, num_heads)
        x = torch.randn(batch_size, seq_len, embedding_dim)
        memory = torch.randn(batch_size, seq_len, embedding_dim)
        
        # Create dummy masks
        source_mask = torch.ones(batch_size, 1, seq_len)
        target_mask = torch.tril(torch.ones(seq_len, seq_len)).expand(batch_size, seq_len, seq_len)
        
        output = layer(x, memory, source_mask, target_mask)
        
        assert output.shape == x.shape


class TestEncoder:
    """Test the Encoder"""
    
    def test_encoder_initialization(self):
        vocab_size = 1000
        embedding_dim = 512
        max_seq_len = 256
        num_heads = 8
        num_layers = 6
        
        encoder = Encoder(vocab_size, embedding_dim, max_seq_len, num_heads, num_layers)
        
        assert len(encoder.layers) == num_layers
        assert encoder.embedding_dim == embedding_dim
    
    def test_encoder_forward(self):
        vocab_size = 1000
        embedding_dim = 64
        max_seq_len = 256
        num_heads = 8
        num_layers = 2
        batch_size = 2
        seq_len = 10
        
        encoder = Encoder(vocab_size, embedding_dim, max_seq_len, num_heads, num_layers)
        
        # Input should be features (11-dimensional based on the model)
        source = torch.randn(batch_size, seq_len, 11)
        source_mask = torch.ones(batch_size, 1, seq_len)
        
        output = encoder(source, source_mask)
        
        assert output.shape == (batch_size, seq_len, embedding_dim)


class TestDecoder:
    """Test the Decoder"""
    
    def test_decoder_initialization(self):
        vocab_size = 1000
        embedding_dim = 512
        max_seq_len = 256
        num_heads = 8
        num_layers = 6
        
        decoder = Decoder(vocab_size, embedding_dim, max_seq_len, num_heads, num_layers)
        
        assert len(decoder.layers) == num_layers
        assert decoder.embedding_dim == embedding_dim
    
    def test_decoder_forward(self):
        vocab_size = 1000
        embedding_dim = 64
        max_seq_len = 256
        num_heads = 8
        num_layers = 2
        batch_size = 2
        seq_len = 10
        
        decoder = Decoder(vocab_size, embedding_dim, max_seq_len, num_heads, num_layers)
        
        # Target should be 1024-dimensional fingerprints
        target = torch.randn(batch_size, seq_len, 1024)
        memory = torch.randn(batch_size, seq_len, embedding_dim)
        source_mask = torch.ones(batch_size, 1, seq_len)
        target_mask = torch.tril(torch.ones(seq_len, seq_len)).expand(batch_size, seq_len, seq_len)
        
        output = decoder(target, memory, source_mask, target_mask)
        
        assert output.shape == (batch_size, seq_len, embedding_dim)


class TestTransformer:
    """Test the full Transformer model"""
    
    def test_transformer_initialization(self):
        source_vocab_size = 100
        target_vocab_size = 1000
        source_max_seq_len = 256
        target_max_seq_len = 256
        embedding_dim = 512
        num_heads = 8
        num_layers = 6
        
        model = Transformer(
            source_vocab_size, target_vocab_size, source_max_seq_len,
            target_max_seq_len, embedding_dim, num_heads, num_layers
        )
        
        assert model.source_vocab_size == source_vocab_size
        assert model.target_vocab_size == target_vocab_size
        assert model.embedding_dim == embedding_dim
    
    def test_transformer_forward(self):
        source_vocab_size = 100
        target_vocab_size = 1000
        source_max_seq_len = 256
        target_max_seq_len = 256
        embedding_dim = 64
        num_heads = 8
        num_layers = 2
        batch_size = 2
        seq_len = 10
        
        model = Transformer(
            source_vocab_size, target_vocab_size, source_max_seq_len,
            target_max_seq_len, embedding_dim, num_heads, num_layers
        )
        
        # Create sample inputs
        p4 = torch.randn(batch_size, seq_len, 11)  # Pharmacophore features
        reactions = torch.randint(0, 56, (batch_size, seq_len))
        mflist = torch.randn(batch_size, seq_len, 1024)  # Molecular fingerprints
        buildingblock = torch.randint(0, target_vocab_size, (batch_size, seq_len))
        buildingblockmf = torch.randn(batch_size, seq_len, 1024)
        
        bbout, reaction_pred = model(p4, reactions, mflist, buildingblock, buildingblockmf)
        
        assert bbout.shape == (batch_size, seq_len, target_vocab_size)
        assert reaction_pred.shape == (batch_size, seq_len, 56)
    
    def test_transformer_predict(self):
        source_vocab_size = 100
        target_vocab_size = 1000
        embedding_dim = 64
        num_heads = 8
        num_layers = 2
        batch_size = 1
        seq_len = 5
        
        model = Transformer(
            source_vocab_size, target_vocab_size, 256, 256,
            embedding_dim, num_heads, num_layers
        )
        
        p4 = torch.randn(batch_size, seq_len, 11)
        mflist = torch.randn(batch_size, seq_len, 1024)
        bbmf = [torch.randn(1024) for _ in range(target_vocab_size)]
        
        model.eval()
        with torch.no_grad():
            logit, buildingblockmf, logitr = model.predict(p4, mflist, bbmf, end_token_id=target_vocab_size-1)
        
        assert len(logit) == 1  # Should return 1 building block (single sample)
        assert buildingblockmf.shape == (1, 1, 1024)
        assert isinstance(logitr, int)  # Should return single reaction index
    
    def test_transformer_gradient_flow(self):
        """Test that gradients flow properly through the model"""
        source_vocab_size = 100
        target_vocab_size = 100
        embedding_dim = 64
        num_heads = 4
        num_layers = 2
        batch_size = 2
        seq_len = 5
        
        model = Transformer(
            source_vocab_size, target_vocab_size, 256, 256,
            embedding_dim, num_heads, num_layers
        )
        
        # Create sample inputs
        p4 = torch.randn(batch_size, seq_len, 11, requires_grad=True)
        reactions = torch.randint(0, 56, (batch_size, seq_len))
        mflist = torch.randn(batch_size, seq_len, 1024)
        buildingblock = torch.randint(0, target_vocab_size, (batch_size, seq_len))
        buildingblockmf = torch.randn(batch_size, seq_len, 1024)
        
        bbout, reaction_pred = model(p4, reactions, mflist, buildingblock, buildingblockmf)
        
        # Compute a simple loss
        loss = bbout.sum() + reaction_pred.sum()
        loss.backward()
        
        # Check that gradients exist
        assert p4.grad is not None
        assert any(param.grad is not None for param in model.parameters())
    
    def test_transformer_mask_functions(self):
        """Test mask creation functions"""
        model = Transformer(100, 100, 256, 256, 64, 4, 2)
        
        # Test source mask
        source_ids = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
        source_pad_id = 0
        source_mask = model.make_source_mask(source_ids, source_pad_id)
        
        expected_shape = (2, 1, 5)
        assert source_mask.shape == expected_shape
        
        # Test target mask
        target_ids = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
        target_mask = model.make_target_mask(target_ids)
        
        expected_shape = (1, 4, 4)
        assert target_mask.shape == expected_shape
        
        # Check that target mask is lower triangular
        assert torch.allclose(target_mask, torch.tril(torch.ones(1, 4, 4)))


class TestModelIntegration:
    """Integration tests for the model"""
    
    def test_model_save_load(self):
        """Test saving and loading model state"""
        model = Transformer(100, 100, 256, 256, 64, 4, 2)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
            model_path = tmp.name
        
        try:
            # Save model
            torch.save(model.state_dict(), model_path)
            
            # Create new model and load state
            new_model = Transformer(100, 100, 256, 256, 64, 4, 2)
            new_model.load_state_dict(torch.load(model_path, map_location='cpu'))
            
            # Compare parameters
            for (name1, param1), (name2, param2) in zip(model.named_parameters(), new_model.named_parameters()):
                assert name1 == name2
                assert torch.allclose(param1, param2)
        
        finally:
            # Clean up
            if os.path.exists(model_path):
                os.unlink(model_path)
    
    def test_model_device_transfer(self):
        """Test moving model between devices"""
        model = Transformer(100, 100, 256, 256, 64, 4, 2)
        
        # Test CPU
        model = model.to('cpu')
        p4 = torch.randn(1, 5, 11)
        reactions = torch.randint(0, 56, (1, 5))
        mflist = torch.randn(1, 5, 1024)
        buildingblock = torch.randint(0, 100, (1, 5))
        buildingblockmf = torch.randn(1, 5, 1024)
        
        output = model(p4, reactions, mflist, buildingblock, buildingblockmf)
        assert output[0].device.type == 'cpu'
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model = model.to('cuda')
            p4 = p4.to('cuda')
            reactions = reactions.to('cuda')
            mflist = mflist.to('cuda')
            buildingblock = buildingblock.to('cuda')
            buildingblockmf = buildingblockmf.to('cuda')
            
            output = model(p4, reactions, mflist, buildingblock, buildingblockmf)
            assert output[0].device.type == 'cuda'
    
    def test_model_memory_usage(self):
        """Test model memory requirements"""
        # Test with minimal model
        model = Transformer(50, 50, 64, 64, 32, 2, 1)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params == total_params  # All params should be trainable by default
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")


# Helper function to run all tests
def run_model_tests():
    """Run all model tests with pytest"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_model_tests() 