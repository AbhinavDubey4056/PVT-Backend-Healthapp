import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC

# Load dataset
data = pd.read_csv("Dataset.csv")

# Split features and target
X = data.drop(columns=['Disease'])
y = data['Disease']

# Simple split train test wali 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Check smallest class size
min_class_size = y_train.value_counts().min()
print(f"Smallest class size = {min_class_size}")

# Base model jo maine use kra h ( u can add mny models )
rf = RandomForestClassifier(random_state=42)

if min_class_size < 2:
    print("\n⚠️ Too few samples per class for GridSearchCV. Skipping cross-validation.")
    
    best_params = {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 2,
        'min_samples_leaf': 1
    }
    rf.set_params(**best_params)
    rf.fit(X_train, y_train)
else:
    print("\n✅ Running GridSearchCV for hyperparameter tuning...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }
    cv_folds = 2 if min_class_size < 3 else 3
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv_folds,
        n_jobs=-1,
        verbose=2,
        scoring='accuracy'
    )
    grid_search.fit(X_train, y_train)
    rf = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)

# Evaluate
y_pred = rf.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(rf, "disease_prediction_best_model.pkl")
print("\n✅ Model saved as 'disease_prediction_best_model.pkl'")
