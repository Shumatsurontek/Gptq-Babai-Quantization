### GPTQ as Babai’s Nearest Plane Algorithm
# ---------------------------------------------------
# This repository contains an implementation and exploration
# of the paper "The Geometry of LLM Quantization: GPTQ as Babai’s Nearest Plane Algorithm" (arXiv:2507.18553).

# Modules:
# - core.py: core GPTQ quantization logic with Babai interpretation
# - lattice_utils.py: tools for lattice reduction (LLL) and Cholesky factorization
# - experiments/: demo on toy model weights and Hessians

# Structure preview below. You can run `pip install -r requirements.txt`
# and then run the demo in experiments/demo.ipynb

# GPTQ = Babai: Quantification par projection sur réseau

Ce repo propose une implémentation de GPTQ comme algorithme de Babai, basé sur le papier [arXiv:2507.18553](https://arxiv.org/abs/2507.18553).

## Contenu
- `core.py`: quantification GPTQ vue comme projection sur réseau
- `lattice_utils.py`: réduction LLL, Cholesky
- `experiments/demo.ipynb`: visualisation + erreurs

## Exécution
```bash
pip install -r requirements.txt
cd experiments
jupyter notebook demo.ipynb
