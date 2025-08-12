# MolDiff Implementation 
## GNN + Diffusion Model for Molecular Generation

## Phase 1: Foundation knowledge (week 1-2)

Papers to read:
1. Denoising Diffusion Probabilistic Models (https://arxiv.org/abs/2006.11239) (Ho et al., 2020)
  - Focus on: forward and reverse processes, loss function, and sampling.
  - Implementation: create simple 1D diffusion model with synthetic data.
2. Improved Denoising Diffusion Probabilistic Models (https://arxiv.org/abs/2102.09672) (Nichol and Dhariwal, 2021)
  - Focus on: noise scheduling, parameterization, and sampling improvements.
  - Implementation: apply improved sampling to the 1D model.
3. MolDiff: Addressing the Atom-Bond Inconsistency Problem in 3D Molecule Diffusion Generation (https://arxiv.org/abs/2305.07508) (Peng et al., 2023)
  - Focus on: graph diffusion formulation, atom-bond consistency, and molecular generation.
  - Implementation: Target paper

Additional papers:
4. Graph Neural Networks: A Review of Methods and Applications (Zhang et al., 2018)
  - Focus on: GNN architectures, applications in chemistry.
5. Molecular Graph Generation with Diffusion Models (Zhang et al., 2023)
  - Focus on: diffusion models for molecular graphs, comparison with other methods.

**[Result of Phase 1:
- Implement basic DDPM in 1D/2D
- Create molecular graph data loaders using DeepChem
- Set up development environment with PyTorch, DeepChem, and RDKit.


## Phase 2: Core Implementation (week 3-4)

#### Week 3: Graph Representation & Data Pipeline

Goal: Handle molecular graphs properly

Tasks:
- implement molecular graph representation using DeepChem.
- Create dataset classes for molecular SMILES -> graph conversion.
- set up atom/bond feature encoding
- test with small molecules first

DeepChem dataset:
- small: deepchem.molnet.load_tox21() (around 8K molecules)
- scale up to: deepchem.molnet.load_zinc15() (around 250K molecules)

#### Week 4: GNN Architecture
Goal: Implement GNN architecture for molecular graphs

Tasks:
- Implement GNN layers (GCN, GAT, etc.) using PyTorch Geometric.
- Add attention mechanisms for better node representation.
- Handle variable graph sizes with proper masking
- Test GNN on simple molecular property prediction]()**

#### Week 5: Diffusion Model Integration
Goal: Adapt DDPM to work with molecular graphs

Tasks:
- Implement forward diffusion process for molecular graphs.
- Handle discrete atom types vs continuous coordinates.
- implenent reverse diffusion process.
- create noise scheduling for molecular graphs

#### Week 6: Training and Evaluation
Goal: Train the complete MolDiff model

Tasks:
- Implement the MolDiff-specific loss function.
- Add atom-bond consistency constraints
- Create training loop with proper validation
- Monitor molecular validity during training


## Phase 3: Advanced Features (week 7-9)

#### Week 7: Chemical Validtity & Constraints
Goal: Ensure generated molecules are chemically valid

Tasks:
- Implement valency constraints
- add chemical rule enforcement
- Create molecular validity metrics using RDKit
- Implement rejection sampling for invalid molecules

#### Week 8: DeepChem Integration
Goal: Make it work seamlessly with DeepChem ecosystem

Tasks:
- Create DeepChem-compatible model classes
- Implement DeepChem featurizers integration
- Add molecular property evaluation using DeepChem models
- Create save/load functionality compatible with DeepChem

#### Week 9: Evaluation & Benchmarking
Goal: Evaluate MolDiff performance

Tasks:
- Create evaluation metrics for molecular generation (validity, diversity, novelty).
- Benchmark against existing methods (e.g., GraphVAE, MolGAN, CVAE).
- Create property-conditioned generation.
- Generate evaluation report with plots and statistics

Evaluation metrics:
- Validity: percentage of valid molecules
- Uniqueness: percentage of unique molecules
- Novelty: percentage of novel molecules (not in training set)
- Property distribution: how well properties match training data
- Molecular diversity: diversity of generated molecules

## Phase 4: Finalization and Documentation (week 10)

Code quality and testing:
- unit tests for core components (GNN, diffusion process, data pipeline)
- integration tests with DeepChem
- code refactoring
- performance profiling and improvement

Documentation and examples
- Jupyter notebooks with usage examples
- example usage with different dataset
- visualisation tools for gneerated moleculs

Final integration and packaging
- package as pip-installable library
- Github repo with CI/CD setup
- blog post 



## Idea for repo structure

moldiff/
├── README.md
├── setup.py
├── requirements.txt
├── environment.yml
├── .gitignore
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── tests.yml
│
├── moldiff/                          # Main package
│   ├── __init__.py
│   ├── core/                         # Core diffusion components
│   │   ├── __init__.py
│   │   ├── scheduler.py              # DDPMScheduler class
│   │   ├── ddpm.py                   # Main DDPM class
│   │   └── sampling.py               # Sampling algorithms
│   │
│   ├── models/                       # Neural network architectures
│   │   ├── __init__.py
│   │   ├── unet.py                   # U-Net for 1D/2D (Phase 1)
│   │   ├── gnn/                      # GNN components (Phase 2)
│   │   │   ├── __init__.py
│   │   │   ├── layers.py             # GNN layers (GCN, GAT, etc.)
│   │   │   ├── attention.py          # Attention mechanisms
│   │   │   └── molecular_gnn.py      # Molecule-specific GNN
│   │   └── diffusion_gnn.py          # Combined GNN+Diffusion (Phase 2)
│   │
│   ├── data/                         # Data handling
│   │   ├── __init__.py
│   │   ├── loaders.py                # Dataset classes
│   │   ├── molecular/                # Molecular data specific
│   │   │   ├── __init__.py
│   │   │   ├── featurizers.py        # Atom/bond feature encoding
│   │   │   ├── graph_utils.py        # SMILES -> graph conversion
│   │   │   └── deepchem_integration.py
│   │   └── synthetic.py              # Toy datasets (Phase 1)
│   │
│   ├── training/                     # Training components
│   │   ├── __init__.py
│   │   ├── trainer.py                # Main training class
│   │   ├── losses.py                 # Loss functions
│   │   └── callbacks.py              # Training callbacks
│   │
│   ├── evaluation/                   # Evaluation and metrics
│   │   ├── __init__.py
│   │   ├── metrics.py                # Molecular metrics (validity, etc.)
│   │   ├── benchmarks.py             # Benchmark comparisons
│   │   └── visualization.py          # Plotting and visualization
│   │
│   ├── chemistry/                    # Chemical constraints & validity
│   │   ├── __init__.py
│   │   ├── constraints.py            # Valency, chemical rules
│   │   ├── validity.py               # Molecular validity checks
│   │   └── rdkit_utils.py            # RDKit integration
│   │
│   └── utils/                        # Utilities
│       ├── __init__.py
│       ├── config.py                 # Configuration management
│       ├── logging.py                # Logging setup
│       └── io.py                     # Save/load utilities
│
├── experiments/                      # Experiment configurations
│   ├── configs/                      # YAML/JSON config files
│   │   ├── phase1_1d_ddpm.yaml
│   │   ├── phase2_molecular_gnn.yaml
│   │   └── phase3_moldiff.yaml
│   ├── scripts/                      # Training scripts
│   │   ├── train_phase1.py
│   │   ├── train_molecular_gnn.py
│   │   └── train_moldiff.py
│   └── results/                      # Experiment outputs
│       ├── phase1/
│       ├── phase2/
│       └── phase3/
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_foundation_ddpm.ipynb      # Phase 1 exploration
│   ├── 02_molecular_graphs.ipynb    # Graph representation
│   ├── 03_gnn_exploration.ipynb     # GNN architectures
│   ├── 04_diffusion_gnn.ipynb       # Combined model
│   ├── 05_evaluation.ipynb          # Results analysis
│   └── examples/                     # Usage examples
│       ├── quickstart.ipynb
│       ├── custom_datasets.ipynb
│       └── property_conditioning.ipynb
│
├── tests/                            # Unit and integration tests
│   ├── __init__.py
│   ├── conftest.py                   # Pytest configuration
│   ├── unit/                         # Unit tests
│   │   ├── test_scheduler.py
│   │   ├── test_ddpm.py
│   │   ├── test_models.py
│   │   ├── test_data_loaders.py
│   │   └── test_gnn_layers.py
│   ├── integration/                  # Integration tests
│   │   ├── test_training_pipeline.py
│   │   ├── test_deepchem_integration.py
│   │   └── test_end_to_end.py
│   └── fixtures/                     # Test data and fixtures
│       ├── sample_molecules.smi
│       └── test_configs.yaml
│
├── docs/                             # Documentation
│   ├── source/
│   │   ├── conf.py
│   │   ├── index.rst
│   │   ├── api.rst
│   │   ├── tutorials/
│   │   └── examples/
│   └── build/
│
├── scripts/                          # Utility scripts
│   ├── setup_environment.py
│   ├── download_data.py
│   └── benchmark.py
│
└── assets/                           # Static assets
    ├── images/
    ├── plots/
    └── paper_figures/

