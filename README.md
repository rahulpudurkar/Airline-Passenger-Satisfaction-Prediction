# Airline Passenger Satisfaction Prediction

This project aims to predict airline passenger satisfaction using logistic regression models. The dataset is based on an airline survey, and the primary objective is to classify passengers as either satisfied or unsatisfied with their flight experience. The project also explores the use of multinomial logistic regression and K-nearest neighbors for predicting passenger class (business, economy, or economy plus). Finally, logistic regression with L1 and L2 penalties is employed to further refine the model.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Data Overview](#data-overview)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [Contributing](#contributing)

## Introduction

This project analyzes airline passenger satisfaction and flight class prediction using machine learning models. The dataset contains various demographic, flight-related, and survey feedback information about airline passengers. Using logistic regression, the project aims to predict whether a passenger is satisfied based on these features. Furthermore, it also involves multinomial logistic regression and K-nearest neighbors to predict the flight class of the passengers.

## Features

- **Binary Logistic Regression**: Predicts passenger satisfaction (Satisfied vs. Not Satisfied).
- **Multinomial Logistic Regression**: Predicts the flight class (Business, Economy, or Economy Plus).
- **K-nearest Neighbors**: Another classification method to predict flight class.
- **Regularized Logistic Regression (L1 and L2 penalties)**: Enhances prediction accuracy by penalizing irrelevant features.
- **Probability Prediction**: Provides a probabilistic prediction for passenger satisfaction.

## Installation

To use this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/airline-satisfaction-prediction.git
   cd airline-satisfaction-prediction
   ```

2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the `Invistico_Airline.csv` dataset is in the working directory. You can find this file on Blackboard or from the repository.

## Data Overview

The dataset contains the following columns:
- **satisfaction**: Whether the passenger was satisfied with the flight experience (Yes/No).
- **Class**: The class of flight the passenger took (Business, Economy, Economy Plus).
- **Demographic and Flight-Related Information**: Columns like age, flight distance, seat comfort, and in-flight entertainment.

## Usage

### Example: Logistic Regression for Passenger Satisfaction

1. **Load and inspect the data**:
   ```python
   data = pd.read_csv('Invistico_Airline.csv')
   print(data.head())
   ```

2. **Preprocess the data**:
   - Drop NaN values and irrelevant columns.
   - Perform one-hot encoding for categorical variables.

3. **Train and Evaluate a Logistic Regression Model**:
   ```python
   from sklearn.linear_model import LogisticRegression
   
   model = LogisticRegression()
   model.fit(X_train, y_train)
   
   # Evaluate the model
   accuracy = model.score(X_test, y_test)
   print(f"Test Accuracy: {accuracy}")
   ```

### Multinomial Logistic Regression for Flight Class Prediction
```python
from sklearn.linear_model import LogisticRegression

# Train a multinomial logistic regression model
model_multi = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model_multi.fit(X_train_multi, y_train_multi)

# Test the model
accuracy = model_multi.score(X_test_multi, y_test_multi)
print(f"Test Accuracy: {accuracy}")
```

### K-Nearest Neighbors
```python
from sklearn.neighbors import KNeighborsClassifier

# Train a K-Nearest Neighbors model with 7 neighbors
knn_model = KNeighborsClassifier(n_neighbors=7)
knn_model.fit(X_train_knn, y_train_knn)

# Evaluate the KNN model
accuracy = knn_model.score(X_test_knn, y_test_knn)
print(f"Test Accuracy: {accuracy}")
```

## Models

1. **Binary Logistic Regression**: Used to predict binary outcomes like passenger satisfaction.
2. **Multinomial Logistic Regression**: Used to predict multi-class outcomes such as flight class.
3. **K-Nearest Neighbors**: A classification technique used to predict flight class.
4. **Logistic Regression with Penalties**: Regularized logistic regression with L1 and L2 penalties for improved accuracy.

## Results

- **Binary Logistic Regression**: The model provides predictions on whether a passenger is satisfied with their flight experience, with an accuracy score of over 90% on the test set.
- **Multinomial Logistic Regression**: The model achieves good accuracy in predicting flight class, with opportunities to fine-tune further.
- **K-Nearest Neighbors**: Performed reasonably well, though training time increased due to the dataset size.
- **Regularized Logistic Regression**: L2 penalty slightly improved test accuracy, while L1 penalty showed some feature reduction with near-zero coefficients.

## Contributing

We welcome contributions! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

---

