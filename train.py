import torch 
from Dataloader import Datasetp4,custom_collate_fn
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from model import Transformer
import tqdm
import torch.nn.functional as F


def random_rotation_matrices(batch_size):
    # Generate random angles
    theta_x = torch.rand(batch_size) * 2 * torch.pi
    theta_y = torch.rand(batch_size) * 2 * torch.pi
    theta_z = torch.rand(batch_size) * 2 * torch.pi

    # Rotation matrices around the x-axis
    Rx = torch.zeros((batch_size, 3, 3))
    Rx[:, 0, 0] = 1
    Rx[:, 1, 1] = torch.cos(theta_x)
    Rx[:, 1, 2] = -torch.sin(theta_x)
    Rx[:, 2, 1] = torch.sin(theta_x)
    Rx[:, 2, 2] = torch.cos(theta_x)

    # Rotation matrices around the y-axis
    Ry = torch.zeros((batch_size, 3, 3))
    Ry[:, 0, 0] = torch.cos(theta_y)
    Ry[:, 0, 2] = torch.sin(theta_y)
    Ry[:, 1, 1] = 1
    Ry[:, 2, 0] = -torch.sin(theta_y)
    Ry[:, 2, 2] = torch.cos(theta_y)

    # Rotation matrices around the z-axis
    Rz = torch.zeros((batch_size, 3, 3))
    Rz[:, 0, 0] = torch.cos(theta_z)
    Rz[:, 0, 1] = -torch.sin(theta_z)
    Rz[:, 1, 0] = torch.sin(theta_z)
    Rz[:, 1, 1] = torch.cos(theta_z)
    Rz[:, 2, 2] = 1

    # Combined rotation matrix
    R = torch.bmm(Rz, torch.bmm(Ry, Rx))
    return R

def calculate_building_block_loss(bb_representations, buildingblock, fingerprint_list, model, device):
    """
    Calculate the building block loss using cosine similarity.
    L_B = (1/n) * sum(Z' · Z / (||Z'|| * ||Z||))
    where Z' = W * f_p(B_{i+1}) + b (encoded fingerprint) and Z = building block representations
    """
    batch_size, seq_len, _ = bb_representations.shape
    
    # Get actual fingerprints for building blocks (Z')
    encoded_fingerprints = []
    for i, bb_indices in enumerate(buildingblock):
        batch_fingerprints = []
        for j, bb_idx in enumerate(bb_indices):
            if bb_idx < len(fingerprint_list):
                # Get fingerprint and encode it: Z' = W * f_p(B_{i+1}) + b
                fp = torch.tensor(fingerprint_list[bb_idx], dtype=torch.float32, device=device)
                encoded_fp = model.fingerprint_encoder(fp)
                batch_fingerprints.append(encoded_fp)
            else:
                # Handle special tokens (padding, end tokens, etc.)
                batch_fingerprints.append(torch.zeros(model.embedding_dim, device=device))
        encoded_fingerprints.append(torch.stack(batch_fingerprints))
    
    encoded_fingerprints = torch.stack(encoded_fingerprints)  # [batch_size, seq_len, embedding_dim]
    
    # Calculate cosine similarity for building block loss
    # L_B = (1/n) * sum(Z' · Z / (||Z'|| * ||Z||))
    z_prime = encoded_fingerprints  # Z' 
    z = bb_representations  # Z
    
    # Normalize vectors
    z_prime_norm = F.normalize(z_prime, p=2, dim=-1)
    z_norm = F.normalize(z, p=2, dim=-1)
    
    # Cosine similarity
    cosine_sim = torch.sum(z_prime_norm * z_norm, dim=-1)  # [batch_size, seq_len]
    
    # Average over sequence length and batch
    lossbb = -torch.mean(cosine_sim)  # Negative because we want to maximize similarity
    
    return lossbb

# Load datap4 from a pickle file
with open('datap4.pkl', 'rb') as f:
    datap4 = pickle.load(f)

# Load data from a pickle file
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)


# Initialize the transformer model



list_index=list(range(len(datap4)))

from utils import load_enamine_building_blocks
filtered_cl = load_enamine_building_blocks()

compound_list=filtered_cl

bb= [Chem.MolFromSmiles(m) for m in compound_list]
bbvocablen=len(bb)

train_data, test_data, _, _ = train_test_split(list_index, list_index, test_size=0.05, random_state=42)
train_data, val_data, _, _ = train_test_split(train_data, train_data, test_size=0.1, random_state=42)
print(len(train_data), len(val_data), len(test_data))
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
model.train()

fingerprint_list = []
for item in bb:
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(item, 2, nBits=1024)
    fingerprint_array = np.array(list(fingerprint.ToBitString())).astype(np.float32)
    fingerprint_tensor = torch.from_numpy(fingerprint_array)
    fingerprint_list.append(fingerprint_array)

# fingerprint_tensor = torch.stack(fingerprint_list)
bbvocablen=len(fingerprint_list)
dataset=Datasetp4(data,datap4,train_data,fingerprint_list)
train_load=DataLoader(dataset,collate_fn=custom_collate_fn,batch_size=56)


# Initialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define the number of epochs
num_epochs = 50
vali_data=Datasetp4(data,datap4,val_data,fingerprint_list)
val_load=DataLoader(vali_data,collate_fn=custom_collate_fn,batch_size=46)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model and data to GPU if available
model.to(device)


# Training loop
for epoch in range(num_epochs):
    total_loss = 0.0
    progress_bar = tqdm.tqdm(train_load, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    for i, batch in enumerate(progress_bar):
        # print(i)
        
        if i ==len(progress_bar)-1:
            continue
        p4 = batch[0].to(device)
        batch_size, num_points, _ = p4.shape

        R = random_rotation_matrices(batch_size).to(device)
        # print(batch[3].shape[1] , batch[2].shape[1])
        if batch[3].shape[1] != batch[1].shape[1]:
            continue
        # print("passed")
        p4[:,:,-3:]=torch.bmm(p4[:,:,-3:], R)
        reactions = batch[1].to(device)
        mflist = batch[2].to(device)
        buildingblock = batch[3].to(device)
        buildingblockmf = batch[4].to(device)
        optimizer.zero_grad()
        bbout, reaction_pred, bb_representations = model(p4, reactions, mflist, buildingblock, buildingblockmf)
        
        # Building Block Loss (L_B): Cosine similarity between Z and Z'
        lossbb = calculate_building_block_loss(bb_representations, buildingblock, fingerprint_list, model, device)

        # Reaction Loss (L_rxn): Cross entropy (unchanged)
        reaction_flat = reaction_pred.permute(0, 2, 1)
        lossreactions = F.cross_entropy(reaction_flat, reactions.long())
        
        # Final loss: L = L_B + L_rxn
        loss = lossbb + lossreactions
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"Loss": total_loss / (i+1)})
    progress_bart = tqdm.tqdm(val_load, desc=f" Val Epoch {epoch+1}/{num_epochs}", leave=False)
    val_loss = 0.0
    for i, batch in enumerate(progress_bart):
        
        if i ==len(progress_bart)-1:
            continue
        if batch[3].shape[1] != batch[1].shape[1]:
            continue
        
        p4 = batch[0].to(device)
        batch_size, num_points, _ = p4.shape
        
        R = random_rotation_matrices(batch_size).to(device)
        p4[:,:,-3:]=torch.bmm(p4[:,:,-3:], R)
        reactions = batch[1].to(device)
        mflist = batch[2].to(device)
        
        buildingblock = batch[3].to(device)
        buildingblockmf = batch[4].to(device)
        
        
        
        optimizer.zero_grad()
        
        bbout, reaction_pred, bb_representations = model(p4, reactions, mflist, buildingblock, buildingblockmf)
        
        # Building Block Loss (L_B): Cosine similarity between Z and Z'
        lossbb = calculate_building_block_loss(bb_representations, buildingblock, fingerprint_list, model, device)

        # Reaction Loss (L_rxn): Cross entropy (unchanged)
        reaction_flat = reaction_pred.permute(0, 2, 1)
        lossreactions = F.cross_entropy(reaction_flat, reactions.long())
        
        # Final loss: L = L_B + L_rxn
        loss = lossbb + lossreactions
        
        val_loss += loss.item()
        progress_bar.set_postfix({"Val Loss": val_loss / (i+1)})
        
    progress_bar.close()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_load)}, =Val Loss: {val_loss / len(val_load)}")
# Save the model
torch.save(model.state_dict(), 'modele.pth')
