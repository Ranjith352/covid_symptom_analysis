import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
df = pd.read_csv('covid_symptom_dataset.csv')

# Display basic info
print("Dataset Shape:", df.shape)
print(df.head())

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Convert categorical columns to numerical values
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Test_Result'] = df['Test_Result'].map({'Positive': 1, 'Negative': 0})
df['Severity'] = df['Severity'].map({'Asymptomatic': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3})

# Separate features (X) and target (y)
X = df.drop(['Severity', 'Test_Result', 'Total_Symptoms'], axis=1)  # Drop target & redundant feature
y = df['Severity']  # Target variable

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Handle Class Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Display data shape after SMOTE
print("Training Data Shape After SMOTE:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Initialize Decision Tree with optimized parameters
dt_model = DecisionTreeClassifier(max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
dt_model.fit(X_train, y_train)

# Predictions & Evaluation for Decision Tree
y_pred_dt = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)
print(f"\nDecision Tree Accuracy: {dt_accuracy * 100:.2f}%")
print("Classification Report (Decision Tree):\n", classification_report(y_test, y_pred_dt))

# Perform 5-fold Cross-Validation
dt_cv_scores = cross_val_score(dt_model, X, y, cv=5, scoring='accuracy')
print(f"Cross-validation scores (DT): {dt_cv_scores}")
print(f"Mean accuracy (DT): {dt_cv_scores.mean() * 100:.2f}%")

# Hyperparameter Tuning with Grid Search (Optional)
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best Decision Tree Model
best_dt_model = grid_search.best_estimator_
best_dt_accuracy = best_dt_model.score(X_test, y_test)
print(f"Best DT Parameters: {grid_search.best_params_}")
print(f"Test Accuracy with Best DT Model: {best_dt_accuracy * 100:.2f}%")

# Random Forest Classifier (Alternative Model)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"\nRandom Forest Accuracy: {rf_accuracy * 100:.2f}%")
print("Classification Report (Random Forest):\n", classification_report(y_test, y_pred_rf))

# Feature Importance Analysis
features = X.columns
importances = best_dt_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Print feature rankings
print("\nFeature Ranking:")
for i in range(len(features)):
    print(f"{i + 1}. Feature: {features[indices[i]]}, Importance: {importances[indices[i]]}")

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importance (Decision Tree)")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), features[indices], rotation=90)
plt.tight_layout()
plt.show()

# Save the best model and features
joblib.dump(best_dt_model, 'model_dt.joblib')
joblib.dump(rf_model, 'model_rf.joblib')
joblib.dump(X.columns, 'features.joblib')

print("Models and feature names saved successfully.")
