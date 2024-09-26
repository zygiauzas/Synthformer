import torch
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable
from egnn_pytorch import EGNN

# Embedding the input sequence
class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)

# The positional encoding vector
class PositionalEncoder(nn.Module):
    def __init__(self, embedding_dim, max_seq_length=512, dropout=0.1):
        super(PositionalEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_length, embedding_dim)
        for pos in range(max_seq_length):
            for i in range(0, embedding_dim, 2):
                pe[pos, i] = math.sin(pos/(10000**(2*i/embedding_dim)))
                pe[pos, i+1] = math.cos(pos/(10000**((2*i+1)/embedding_dim)))
        pe = pe.unsqueeze(0)        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x*math.sqrt(self.embedding_dim)
        seq_length = x.size(1)
        pe = Variable(self.pe[:, :seq_length], requires_grad=False).to(x.device)
        # Add the positional encoding vector to the embedding vector
        x = x + pe
        x = self.dropout(x)
        return x

# Self-attention layer
class SelfAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        key_dim = key.size(-1)
        attn = torch.matmul(query / np.sqrt(key_dim), key.transpose(2, 3))
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(torch.softmax(attn, dim=-1))
        output = torch.matmul(attn, value)

        return output
        
# Multi-head attention layer
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.self_attention = SelfAttention(dropout)
        # The number of heads
        self.num_heads = num_heads
        # The dimension of each head
        self.dim_per_head = embedding_dim // num_heads
        # The linear projections
        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, query, key, value, mask=None):
        # Apply the linear projections
        batch_size = query.size(0)
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)
        # Reshape the input
        query = query.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        # Calculate the attention
        scores = self.self_attention(query, key, value, mask)
        # Reshape the output
        output = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_dim)
        # Apply the linear projection
        output = self.out(output)
        return output

# Norm layer
class Norm(nn.Module):
    def __init__(self, embedding_dim):
        super(Norm, self).__init__()
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        return self.norm(x)


# Transformer encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim=2048, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = Norm(embedding_dim)
        self.norm2 = Norm(embedding_dim)

    def forward(self, x, mask=None):
        x2 = self.norm1(x)
        # Add and Muti-head attention
        x = x + self.dropout1(self.self_attention(x2, x2, x2, mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.feed_forward(x2))
        return x

# Transformer decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim=2048, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.encoder_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = Norm(embedding_dim)
        self.norm2 = Norm(embedding_dim)
        self.norm3 = Norm(embedding_dim)

    def forward(self, x, memory, source_mask, target_mask):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.self_attention(x2, x2, x2, target_mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.encoder_attention(x2, memory, memory, source_mask))
        x2 = self.norm3(x)
        x = x + self.dropout3(self.feed_forward(x2))
        return x

# Encoder transformer
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_len, num_heads, num_layers, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.inputlayer = nn.Linear(11, embedding_dim)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.layers = nn.ModuleList([EncoderLayer(embedding_dim, num_heads, 2048, dropout) for _ in range(num_layers)])
        self.norm = Norm(embedding_dim)
        self.position_embedding = PositionalEncoder(embedding_dim, max_seq_len, dropout)
    
    def forward(self, source, source_mask):
        # Embed the source
        x=self.inputlayer(source)
        # x = self.embedding(source)
        # Add the position embeddings
        # x = self.position_embedding(x)
        # Propagate through the layers
        for layer in self.layers:
            x = layer(x, source_mask)
        # Normalize
        x = self.norm(x)
        return x

# Decoder transformer
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_len,num_heads, num_layers, dropout=0.1):
        super(Decoder, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.inputlayer=nn.Linear(1024, embedding_dim)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, num_heads, 2048, dropout) for _ in range(num_layers)])
        self.norm = Norm(embedding_dim)
        self.position_embedding = PositionalEncoder(embedding_dim, max_seq_len, dropout)
    
    def forward(self, target, memory, source_mask, target_mask):
        # Embed the source
        x = self.inputlayer(target)
        # Add the position embeddings
        x = self.position_embedding(x)
        # Propagate through the layers
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        # Normalize
        x = self.norm(x)
        return x


# Transformers
class Transformer(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, source_max_seq_len, target_max_seq_len, embedding_dim, num_heads, num_layers, dropout=0.1):
        super(Transformer, self).__init__()
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.source_max_seq_len = source_max_seq_len
        self.target_max_seq_len = target_max_seq_len
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.softm=nn.Softmax()

        self.bblinear=nn.Linear(embedding_dim,target_vocab_size)
        self.layer1 = EGNN(dim = 7)
        self.layer2 = EGNN(dim = 7)
        self.layer3 = EGNN(dim = 7)
        self.layer4 = EGNN(dim = 7)
        self.layer5 = EGNN(dim = 7)
      

        self.trans = nn.Linear(7, embedding_dim)
    
        self.encoder = Encoder(source_vocab_size, embedding_dim, source_max_seq_len, num_heads, num_layers, dropout)
        self.decoder = Decoder(target_vocab_size, embedding_dim, target_max_seq_len, num_heads, num_layers, dropout)
        self.final_linear = nn.Linear(1024+target_vocab_size, 56)
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(target_vocab_size, 1024)
    
    def forward(self, p4, reactions, mflist,  buildingblock,buildingblockmf):
        # Encoder forward pass
        source_mask = torch.all(p4 == 0, dim=-1).unsqueeze(-2)
        # print(self.shape)
        # memory = self.encoder(p4, source_mask)
    
        memory,c = self.layer1(p4[:,:,:-4], p4[:,:,-3:])
        memory,c = self.layer2(memory,c )
        memory,c = self.layer3(memory,c )
        memory,c = self.layer4(memory,c )
        memory,c = self.layer5(memory,c )


        memory=self.trans(memory)

        # Decoder forward pass
        batch_size, len_target,_ = mflist.size()
        target_mask = (1 - torch.triu(torch.ones((1, len_target, len_target), device=mflist.device), diagonal=1)).bool()
        mflist=mflist-torch.mean(mflist)
        output = self.decoder(mflist, memory, source_mask, target_mask)
        output = self.dropout(output)
        buildingblockl= self.bblinear(output)
        # print(buildingblockl.shape, buildingblockmf.shape)
        x = self.embedding(buildingblock.long())
        # buildingblockmf=buildingblockmf-torch.mean(buildingblockmf)
        buildingblockmf=x
        merged_tensor = torch.cat((buildingblockl, buildingblockmf), dim=-1)
        

        
        # Final linear layer
        # output = self.dropout(merged_tensor)
        reaction = self.final_linear(merged_tensor)
        
        return buildingblockl,reaction
    def predict(self, p4, mflist,bbmf):
        # Encoder forward pass
        source_mask = torch.all(p4 == 0, dim=-1).unsqueeze(-2)
        # print(self.shape)
        # memory = self.encoder(p4, source_mask)
        # print(p4.shape)
        # p4.squeeze_(0)
        # print(p4.shape)
        memory,c = self.layer1(p4[:,:,:-4], p4[:,:,-3:])
        memory,c = self.layer2(memory,c )
        memory,c = self.layer3(memory,c )
        memory,c = self.layer4(memory,c )
        memory,c = self.layer5(memory,c )
        memory=self.trans(memory)


        # Decoder forward pass
        batch_size, len_target,_ = mflist.size()
        target_mask = (1 - torch.triu(torch.ones((1, len_target, len_target), device=mflist.device), diagonal=1)).bool()
        mflist=mflist-torch.mean(mflist)
        output = self.decoder(mflist, memory, source_mask, target_mask)
        output = self.dropout(output)
        buildingblockl= self.bblinear(output)
        probbb=self.softm(buildingblockl)
    
        
        logit=torch.multinomial(probbb[:,-1,:].squeeze(), 2)
        if logit[0]<len(bbmf):
            buildingblockmf=self.embedding(logit[0]).unsqueeze(0).unsqueeze(0)
            print(buildingblockmf.shape)
        else:
            buildingblockmf=self.embedding(logit[0]-10).unsqueeze(0).unsqueeze(0)
        x = self.embedding(logit[0])
        # buildingblockmf=buildingblockmf-torch.mean(buildingblockmf)
        buildingblockmf=x.unsqueeze(0).unsqueeze(0)
        print(buildingblockmf.shape)
        merged_tensor = torch.cat((buildingblockl[:,-1,:], buildingblockmf[:,-1,:]), dim=-1)

        
        # Final linear layer
        # output = self.dropout(merged_tensor)
        reaction = self.final_linear(merged_tensor)
        probr=self.softm(reaction)
        # print(probr.shape)
        logitr=torch.multinomial(probr[-1,:].squeeze(), 20).tolist()
        return logit,buildingblockmf,logitr
    
    def make_source_mask(self, source_ids, source_pad_id):
        return (source_ids != source_pad_id).unsqueeze(-2)

    def make_target_mask(self, target_ids):
        batch_size, len_target = target_ids.size()
        subsequent_mask = (1 - torch.triu(torch.ones((1, len_target, len_target), device=target_ids.device), diagonal=1)).bool()
        return subsequent_mask