import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


def preprocess_data(data, target_column):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return X, y


def train_and_evaluate_models(X, y):
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Dictionary to hold models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'SVM': SVC(kernel='linear'),
        'Gradient Boosting': GradientBoostingClassifier(),
        'Logistic Regression': LogisticRegression(),
        'k-NN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
    }

    # Train and evaluate models
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"Model: {name}")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\n")

    # Neural Network
    y_train_encoded = to_categorical(y_train)
    y_test_encoded = to_categorical(y_test)
    model = Sequential([
        Dense(64, input_dim=X_train.shape[1], activation='relu'),
        Dense(32, activation='relu'),
        Dense(y_train_encoded.shape[1], activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train_encoded, epochs=50, batch_size=10, verbose=0)
    loss, accuracy = model.evaluate(X_test, y_test_encoded, verbose=0)
    print("Model: Neural Network")
    print("Accuracy:", accuracy)


def train_model(data, target_column, model_type='CART', random_state=42):
    """
    Train a decision tree model using the specified type on the provided data.

    Parameters:
    - data: DataFrame, the preprocessed dataset
    - target_column: str, the name of the target column in the dataset
    - model_type: str, type of the model ('CART' for Gini or 'C4.5' for entropy)
    - random_state: int, seed for the random number generator

    Returns:
    - model: DecisionTreeClassifier, trained model
    - X_test: DataFrame, test features
    - y_test: Series, true values for the test set
    - y_pred: Series, predicted values on the test set
    """
    if data is None:
        print("No data available for model training.")
        return None, None, None, None

    if target_column not in data.columns:
        print("Target column not found in the data.")
        return None, None, None, None

    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Correct the test_size parameter
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

    criterion = 'gini' if model_type == 'CART' else 'entropy'
    model = DecisionTreeClassifier(criterion=criterion, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"Model training completed using {model_type}.")
    return model, X_test, y_test, y_pred


def evaluate_model(y_test, y_pred):
    """
    Evaluate the model's performance using classification report and confusion matrix.

    Parameters:
    - y_test: Series, true labels for the test data
    - y_pred: Series, model's predictions on the test data
    """
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
