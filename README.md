
---

# Loan Approval Prediction Model

## Project Overview

This project involves developing a machine learning model to predict loan approval decisions based on applicant data. The model uses Scikit-learn's `DecisionTreeClassifier` for building and training the predictive model. The goal is to automate and improve the accuracy of loan approval processes.

## Features

- Data preprocessing and feature engineering
- Model training and evaluation using `DecisionTreeClassifier`
- Model serialization with Joblib for deployment
- Real-time loan approval predictions

## Demo of the app : https://huggingface.co/spaces/KsAech/LoanClassifier

## Getting Started

### Prerequisites

To run this project, you need the following Python packages:
- Python 3.10 or higher
- NumPy
- Scikit-learn 1.2.2
- Joblib

You can install the necessary packages using `pip`:

```bash
pip install numpy==1.23.5 scikit-learn==1.2.2 joblib
```

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Ks-Aech/LoanPredictorApp.git
    cd loan-approval-prediction
    ```

### Usage

1. Load the trained model:

    ```python
    import joblib

    # Load the pre-trained model
    model = joblib.load('loan_classifier.joblib')
    ```

2. Use the model to make predictions:
    ```python
    # Example input data
    applicant_data = [[...]]  # Replace with actual data

    # Predict loan approval
    prediction = model.predict(applicant_data)
    print("Loan approval prediction:", prediction)
    ```

### Example

Here's an example of how to use the model in a script:

```python
import joblib

# Load the trained model
model = joblib.load('loan_classifier.joblib')

# Example input data (replace with actual data)
applicant_data = [[25, 50000, 600, 1, 0, 0, 240, 1]]

# Predict loan approval
prediction = model.predict(applicant_data)
print("Loan approval prediction:", prediction)
```

## Project Structure

```
loan-approval-prediction/
├── app.py
├── std_sclaer.bin
├── loan_classifier.joblib
├── requirements.txt
└── README.md
```

- `app.py`: Main script to load and use the model.
- `loan_classifier.joblib`: Serialized machine learning model.
- `requirements.txt`: List of required Python packages.
- `README.md`: Project documentation.
- `std_scaler.bi`: Standar scaler used in the model development

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Scikit-learn documentation: https://scikit-learn.org/stable/documentation.html
- NumPy documentation: https://numpy.org/doc/

---

