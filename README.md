# Synthformer 

AI-driven molecular design using transformer neural networks and pharmacophore-based representations.

## Prerequisites

### Required Datasets

**Enamine Building Blocks Database:**
- Register at [Enamine website](https://enamine.net/)
- Download the building blocks database
- Save as `Enamine_Rush-Delivery_Building_Blocks-US_251222cmpd_20250111.sdf`

**PDBbind Dataset (for evaluation):**
- Download from [PDBbind website](http://www.pdbbind.org.cn/)
- Required for model evaluation and validation
- All evaluation in the paper was performed on this dataset
- Follow PDBbind registration and download procedures

Install dependencies:
```bash
conda create --name synthformer --file requirements.txt
conda activate synthformer
```

## Quick Start Workflow

Run these commands in order:

```bash
# 1. Generate molecular data
python generate_data.py

# 2. Generate pharmacophore features  
python generate_p4.py

# 3. Run tests
python test_integration.py
python test_model.py
python test_training.py
python test_dataloader_extended.py
python test_utils.py

# 4. Train model
python train.py

# 5. Run inference (choose one)
python inference.py                    # Standard generation
python inference_pdb_bind.py          # PDBbind targets
python inference_hit_expansion.py     # Hit expansion
python molecule\ optimisation.py      # Multi-objective optimization

# 6. Dock and analyze
python dock_experiments.py --mode standard
python analyze_experiments.py --mode docking
```

## Core Files

- `generate_data.py` - Process building blocks and reactions → `data.pkl`
- `generate_p4.py` - Extract pharmacophore features → `datap4.pkl`  
- `train.py` - Train transformer model → `modele.pth`
- `inference*.py` - Generate molecules
- `dock_experiments.py` - Molecular docking
- `analyze_experiments.py` - Results analysis

## Output

- Trained model: `modele.pth`
- Generated molecules: Various CSV/pickle files
- Docking results and analysis reports 