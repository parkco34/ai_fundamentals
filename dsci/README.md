# Decision Trees Analysis Project

## Overview
This project implements Decision Tree classification on the Iris dataset, featuring hyperparameter tuning through Random Search and detailed error analysis. The implementation uses scikit-learn and includes comprehensive model evaluation.

## Features
- Dataset preparation and preprocessing
- Hyperparameter tuning using RandomizedSearchCV
- Error analysis and misclassification identification
- Visualization of results using pandas DataFrames

## Requirements
```
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.20.0
```

## Project Structure
```
decision_trees/
│
├── src/
│   ├── __init__.py
│   ├── data_preparation.py    # Data loading and preprocessing
│   ├── model_training.py      # Decision Tree implementation
│   └── error_analysis.py      # Error analysis functions
│
├── notebooks/
│   └── analysis.ipynb         # Interactive analysis notebook
│
├── requirements.txt
└── README.md
```

## Installation
1. Clone this repository:
```bash
git clone <repository-url>
cd decision-trees
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage
### Basic Usage
```python
from src.data_preparation import prepare_data
from src.model_training import perform_random_search
from src.error_analysis import analyze_errors

# Prepare data
X_train, X_test, y_train, y_test = prepare_data()

# Perform random search
best_model = perform_random_search(X_train, y_train)

# Analyze errors
misclassified_indices = analyze_errors(best_model, X_test, y_test)
```

### Hyperparameter Tuning
The project implements Random Search for the following hyperparameters:
- max_depth: Range 1-20
- min_samples_split: Range 2-20
- min_samples_leaf: Range 1-10
- criterion: 'gini' or 'entropy'

## Features in Detail

### 1. Data Preparation
- Loads the Iris dataset
- Converts to pandas DataFrame for enhanced visualization
- Splits data into training and testing sets (80/20 split)
- Implements optional data scaling

### 2. Hyperparameter Tuning
- Utilizes RandomizedSearchCV for efficient parameter space exploration
- Implements 5-fold cross-validation
- Provides best parameter combination and model performance metrics

### 3. Error Analysis
- Identifies misclassified instances
- Provides detailed analysis of error patterns
- Includes visualization of misclassification distribution

## Example Output
```
Best Hyperparameters:
- max_depth: 4
- min_samples_split: 5
- min_samples_leaf: 2
- criterion: gini

Model Performance:
- Training Accuracy: 0.975
- Testing Accuracy: 0.967

Misclassified Instances: 3
Indices: [23, 45, 67]
```

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Scikit-learn documentation and contributors
- Iris dataset providers

## Contact
For questions and feedback, please open an issue in the repository.
