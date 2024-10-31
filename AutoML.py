import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

class AutoMLModel:
    def __init__(self, project_name):
        self.project_name = project_name
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None

    def prepare_data(self, data_path, target_column):
        print("Preparing data...")
        self.data = pd.read_csv(data_path)
        self.X = self.data.drop(target_column, axis=1)
        self.y = self.data[target_column]
        print("Data prepared and loaded.")

    def preprocess_data(self):
        print("Preprocessing data...")
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        # Identify numeric and categorical columns
        numeric_features = self.X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = self.X.select_dtypes(include=['object']).columns

        # Create preprocessing steps for numeric and categorical data
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Feature selection
        feature_selector = SelectKBest(f_classif, k='all')

        # Create the preprocessing pipeline
        self.preprocess_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('feature_selector', feature_selector)
        ])

        print("Preprocessing pipeline created.")

    def train_model(self):
        print("Training model...")
        # Define models and their hyperparameters
        models = [
            ('rf', RandomForestClassifier(), {
                'rf__n_estimators': [50, 100, 200],
                'rf__max_depth': [None, 10, 20],
                'preprocessor__feature_selector__k': [5, 10, 'all']
            }),
            ('svm', SVC(), {
                'svm__C': [0.1, 1, 10],
                'svm__kernel': ['rbf', 'linear'],
                'preprocessor__feature_selector__k': [5, 10, 'all']
            })
        ]

        best_score = 0
    
        for name, model, params in models:
            pipeline = Pipeline([
                ('preprocessor', self.preprocess_pipeline),
                (name, model)
        ])
        
        grid_search = GridSearchCV(pipeline, params, cv=5, n_jobs=-1, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)
        
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            self.best_model = grid_search.best_estimator_

        print("Model training completed.")

    def evaluate_model(self):
        print("Evaluating model...")
        y_pred = self.best_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy}")
        print("Classification Report:")
        print(report)

    def deploy_model(self):
        print("Simulating model deployment...")
        # In a real scenario, you would deploy the model to a production environment
        print("Model deployed successfully.")

    def monitor_model(self):
        print("Simulating model monitoring...")
        # In a real scenario, you would set up monitoring for the deployed model
        print("Model monitoring set up successfully.")

    def optimize_model(self):
        print("Simulating model optimization...")
        # In a real scenario, you would periodically retrain and optimize the model
        print("Model optimization completed.")

    def save_model(self, filepath):
        print(f"Saving model to {filepath}...")
        joblib.dump(self.best_model, filepath)
        print("Model saved successfully.")

# Example usage
if __name__ == "__main__":
    automl = AutoMLModel("my_classification_project")
    automl.prepare_data("D:/Files/diabetes.csv", "Outcome")
    automl.preprocess_data()
    automl.train_model()
    automl.evaluate_model()
    automl.save_model("diabetes_model.joblib")  # Save 
    automl.deploy_model()
    automl.monitor_model()
    automl.optimize_model()