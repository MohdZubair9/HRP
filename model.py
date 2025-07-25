

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("hrp_final.xlsx") 
# print("Data shape:", df.shape)
# print("First few rows:\n", df.head())

X = df.drop(columns=["Height","Blood Pressure","Fetal Position","Fetal Movement","Urine Sugar","High Risk Pregnancy","Height_cm","Fetal position"])
y = df["High Risk Pregnancy"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

logreg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(class_weight='balanced'))
])

dt_pipeline = Pipeline([
    ('dt', DecisionTreeClassifier())
])

rf_pipeline = Pipeline([
    ('rf', RandomForestClassifier(class_weight='balanced'))
])

logreg_params = {
    'logreg__C': [0.01, 0.1, 1, 10],
    'logreg__penalty': ['l2'],
    'logreg__solver': ['lbfgs']
}

dt_params = {
    'dt__max_depth': [2, 4, 6, 8, 10, None],
    'dt__min_samples_split': [2, 5, 10]
}

rf_params = {
    'rf__n_estimators': [50, 100, 200],
    'rf__max_depth': [4, 6, 8, None],
    'rf__min_samples_split': [2, 5, 10]
}

print("\n Tuning Logistic Regression...")
logreg_grid = GridSearchCV(logreg_pipeline, logreg_params, cv=5, scoring='accuracy')
logreg_grid.fit(X_train, y_train)

print("Best Logistic Parameters:", logreg_grid.best_params_)
print("Best Logistic Accuracy:", logreg_grid.best_score_)

print("\n Tuning Decision Tree...")
dt_grid = GridSearchCV(dt_pipeline, dt_params, cv=5, scoring='accuracy')
dt_grid.fit(X_train, y_train)

print("Best Decision Tree Parameters:", dt_grid.best_params_)


print("Best Decision Tree Accuracy:", dt_grid.best_score_)

print("\n Tuning Random Forest...")
rf_grid = GridSearchCV(rf_pipeline, rf_params, cv=5, scoring='accuracy')
rf_grid.fit(X_train, y_train)

print("Best Random Forest Parameters:", rf_grid.best_params_)
print("Best Random Forest Accuracy:", rf_grid.best_score_)


models = {
    "Logistic Regression": logreg_grid,
    "Decision Tree": dt_grid,
    "Random Forest": rf_grid
}

test_accuracies = {}

print("\n Model Evaluation on Test Set:")
for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    test_accuracies[name] = acc

    print(f"\n{name}:\nAccuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

plt.figure(figsize=(10, 6))
sns.barplot(x=list(test_accuracies.keys()), y=list(test_accuracies.values()))
plt.title("Test Accuracy of ML Models")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()


best_model_name = max(test_accuracies, key=test_accuracies.get)
best_model_pipeline = models[best_model_name].best_estimator_
# print("feature importances:",best_model_pipeline)
print("best_model_pipeline:", best_model_pipeline)
print("best_model_name:", best_model_name)
print(f"\n Best Performing Model: {best_model_name} with Accuracy: {test_accuracies[best_model_name]:.4f}")

features = X.columns.tolist()
joblib.dump(best_model_pipeline, "best_model_pipeline.pkl")
joblib.dump(features, "features.pkl")
print(" Best model saved as 'best_model_pipeline.pkl'")
