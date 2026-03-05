# Data Loaders

Each loader returns `(train_data, test_data)` — a tuple of two dicts.

## Format

Each data dict has the following keys:
- `'X'` — feature matrix, `np.ndarray` of shape `(N, D)`
- `'y'` — labels, `np.ndarray` of shape `(N,)`, binary (0/1)
- `'feature_names'` — list of feature name strings
- `'costs'` — `(c1, c2)` misclassification costs *(optional)*

## Available Datasets

| File | Dataset |
|------|---------|
| `breast_cancer_W.py` | Wisconsin Breast Cancer |
| `diabetes.py` | Pima Indians Diabetes |
| `heart_disease.py` | Heart Disease (Cleveland) |
| `hepititus.py` | Hepatitis |
| `ionosphere.py` | Ionosphere |
| `banknote.py` | Banknote Authentication |
| `sonar_rocks.py` | Sonar (Rocks vs Mines) |
| `wheat_seeds.py` | Wheat Seeds |
| `abalone_gender.py` | Abalone Gender |
| `cervical_cancer.py` | Cervical Cancer |
| `chonic_kidney_disease.py` | Chronic Kidney Disease |
| `german_credit.py` | German Credit |
| `gaussian.py` | Synthetic Gaussian |
| `sklearn_synthetic.py` | Sklearn synthetic datasets |
| `sklearn_toy.py` | Sklearn toy datasets |
| `mnist.py` | MNIST (binary subset) |
| `MIMIC_III.py` | MIMIC-III (requires local data) |
| `MIMIC_IV.py` | MIMIC-IV (requires local data) |
| `costcla.py` | Cost-sensitive classification datasets |

## Adding a New Loader

1. Create a new file in `deltas/data/loaders/`, e.g. `my_dataset.py`
2. Implement a function that returns `(train_data, test_data)`:

```python
import numpy as np
from sklearn.model_selection import train_test_split
from deltas.misc.use_two import RANDOM_STATE

def my_dataset(test_size=0.2):
    # load / generate X, y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE)

    train_data = {'X': X_train, 'y': y_train, 'feature_names': [...]}
    test_data  = {'X': X_test,  'y': y_test,  'feature_names': [...]}
    return train_data, test_data
```

3. Import it in `deltas/data/loaders/__init__.py` if needed.
