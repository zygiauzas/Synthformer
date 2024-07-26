import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
import seaborn as sns
def calculate_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        descriptors = {
            'MolWt': Descriptors.HeavyAtomMolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'SMILES': smiles
        }
        return descriptors
    except Exception as e:
        print(f"Error processing SMILES '{smiles}': {e}")
        return None
# Load the data.pkl file
data = pd.read_pickle('data.pkl')

# Plot some graphs about molecular properties
smiles = data[1]
data=[]
# print(smiles)
for i,smi in enumerate(smiles):
    descriptors = calculate_descriptors(smiles[i+1])
    if descriptors:
        descriptors['SMILES'] = smi
        data.append(descriptors)
# TODO: Add your code here to plot the graphs

df = pd.DataFrame(data)
# Save df as CSV
df.to_csv('data.csv', index=False)


# Plotting
plt.figure(figsize=(12, 8))
sns.histplot(df['MolWt'], kde=True)
plt.title('Histogram of Molecular Weight')
plt.xlabel('Molecular Weight')
plt.ylabel('Frequency')
plt.savefig('histogram_molecular_weight.png')
plt.show()

plt.figure(figsize=(12, 8))
sns.scatterplot(x='MolWt', y='LogP', data=df)
plt.title('Molecular Weight vs LogP')
plt.xlabel('Molecular Weight')
plt.ylabel('LogP')
plt.savefig('scatter_molwt_logp.png')
plt.show()

sns.pairplot(df, diag_kind='kde')
plt.savefig('pairplot_descriptors.png')
plt.show()

