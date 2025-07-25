# Iris ML Pipeline Example (Template-Based)

This repository demonstrates how to build a minimal, reproducible ML pipeline using the structure inspired by the [CLAIRE python-ml-research-template](https://github.com/CLAIRE-Labo/python-ml-research-template).

It trains a Random Forest classifier on the Iris dataset using modular components for data loading, training, and evaluation.

## 🔧 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the experiment
python experiments/run_experiment.py
```

Model is saved to `models/iris_model.pkl`.

## 📁 Structure

- `configs/` - Configuration YAMLs for reproducibility
- `experiments/` - Scripts to run full experiments
- `src/` - Source code modules: data, training, evaluation
- `models/` - Trained model storage

## ✅ Inspired by

[CLAIRE python-ml-research-template](https://github.com/CLAIRE-Labo/python-ml-research-template)

Happy to have this listed under the template’s example projects if it helps others!
