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
