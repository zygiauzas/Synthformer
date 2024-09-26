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

# Load datap4 from a pickle file
with open('datap4.pkl', 'rb') as f:
    datap4 = pickle.load(f)

# Load data from a pickle file
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)


# Initialize the transformer model



list_index=list(range(len(datap4)))

with open("mcule_bb.smi") as f:
    line = f.readlines(1000000)
    compound_list=[a.split('\t')[0] for a in line]
    filtered_cl=[]
    for mol in compound_list:
        if len(mol)<30:
            filtered_cl.append(mol)

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
        
        bbout, reaction_pred = model(p4, reactions, mflist, buildingblock, buildingblockmf)
        
        bbout_flat = bbout.permute(0, 2, 1)
        reaction_flat = reaction_pred.permute(0, 2, 1)
        # print(reactions)
        lossbb = F.cross_entropy(bbout_flat, buildingblock.long())

        # print(reactions)
        lossreactions = F.cross_entropy(reaction_flat, reactions.long())
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
        
        bbout, reaction_pred = model(p4, reactions, mflist, buildingblock, buildingblockmf)
        
        bbout_flat = bbout.permute(0, 2, 1)
        reaction_flat = reaction_pred.permute(0, 2, 1)
        lossbb = F.cross_entropy(bbout_flat, buildingblock.long())


        lossreactions = F.cross_entropy(reaction_flat, reactions.long())
        loss = lossbb + lossreactions
        
        val_loss += loss.item()
        progress_bar.set_postfix({"Val Loss": val_loss / (i+1)})
        
    progress_bar.close()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_load)}, =Val Loss: {val_loss / len(val_load)}")
# Save the model
torch.save(model.state_dict(), 'modele.pth')
