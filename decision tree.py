# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Loading the Iris dataset
iris = load_iris()

# The features (sepal length, sepal width, petal length, petal width) and target (species)
X = iris.data
y = iris.target

# Checking the shape of the dataset
print(f"Dataset shape: {X.shape}")

# Splitting the dataset into Training set and Test set (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Feature Scaling (Decision Trees are not sensitive to feature scaling, but it's good practice)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Decision Tree model
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Displaying the Confusion Matrix and Accuracy
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
