import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
class Datasetp4(Dataset):
    def __init__(self, data, p4,indexes,bbmf, transform=None):
        self.data = data[0]
        self.smiles = data[1]
        self.imf = data[2]
        self.p4 = p4
        self.indexes = indexes
        self.bb=bbmf

        self.diclen=len(bbmf)
        self.bbmf=bbmf

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        maxlens=[]
        reactions=[0]
        # reactions.append(0)
       
        index=self.indexes[idx-1]
        sample = self.data[index]
        bbmf=self.bbmf
        # smiles = self.smiles[index]
        smiles=[]
        # reactions.append(sample[0])
        # Initialise start token
        maxlens.append(sample[3])
        arr = np.zeros(1024)
        arr[0] = 1
        mflist=[]
        buildingblock=[]
        buildingblockmf=[]
        bblist=[]
        lens=sample[3]
        sample=self.data[index]
        for a in range(sample[3]+1):
            if a==lens:
                mflist.append(self.imf[sample[1]])
                mflist.append(self.bbmf[sample[1]])
                buildingblock.extend([sample[2],sample[1]])
                buildingblockmf.append(self.bbmf[sample[2]])
                buildingblockmf.append(self.bbmf[sample[1]])
                reactions.append(sample[0])
            else:
            
                smiles.append(self.smiles[sample[2]])
                reactions.append(sample[0])
                mflist.append(self.imf[sample[2]])
                buildingblock.append(sample[2])
                
                buildingblockmf.append(self.bbmf[sample[2]])
                sample = self.data[sample[1]]
        
        buildingblockmf.append(np.zeros(1024))
        mflist.append(arr)
        pp4=self.p4[index]
        # print(pp4)
        reactions.append(0)
        reactions.reverse()
        mflist.reverse()
        buildingblock.reverse()
        buildingblockmf.reverse()
        buildingblock.append(self.diclen+1)
        
        p4T=torch.Tensor(pp4)
        reactionsT=torch.Tensor(reactions)
        mflistT=torch.Tensor(mflist)
        buildingblockT=torch.Tensor(buildingblock)
        buildingblockmfT=torch.Tensor(buildingblockmf)
       
        return p4T, reactionsT, mflistT, buildingblockT,buildingblockmfT, maxlens,self.smiles[index]
    
def custom_collate_fn(batch):
    # Assuming each element in "batch" is a tuple (data, label)
    p4,reactions, mflist, buildingblock,buildingblockmfT,maxlens,smiles = zip(*batch)
    
    # Pad the data sequences
    p4_padded = pad_sequence(p4, batch_first=True, padding_value=0)
    reactions_padded = pad_sequence(reactions, batch_first=True, padding_value=0)
    mflist_padded = pad_sequence(mflist, batch_first=True, padding_value=0)
    buildingblock_padded = pad_sequence(buildingblock, batch_first=True, padding_value=0)
    buildingblockmf_padded = pad_sequence(buildingblockmfT, batch_first=True, padding_value=0)
    # Convert labels to a tensor
    
    
    return p4_padded, reactions_padded, mflist_padded, buildingblock_padded,buildingblockmf_padded,maxlens,smiles 