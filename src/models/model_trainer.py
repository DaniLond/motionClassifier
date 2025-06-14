from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder  
import joblib
import pandas as pd
import numpy as np


class ActivityClassifierTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.label_encoder = LabelEncoder()  
        self.feature_columns = None

    def train_multiple_models(self, X, y):
        """Entrena múltiples modelos y selecciona el mejor"""

       
        print("Codificando etiquetas...")
        y_encoded = self.label_encoder.fit_transform(y)
        print(f"Clases originales: {list(self.label_encoder.classes_)}")
        print(f"Clases codificadas: {np.unique(y_encoded)}")

        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

       
        model_configs = {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'SVM': {
                'model': SVC(random_state=42),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01],
                    'kernel': ['rbf', 'poly']
                }
            },
            'XGBoost': {
               
                'model': XGBClassifier(random_state=42, eval_metric='mlogloss'),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                }
            }
        }

        results = {}
        for name, config in model_configs.items():
            print(f"\nEntrenando {name}...")

            
            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X_train, y_train)

           
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)

           
            y_test_labels = self.label_encoder.inverse_transform(y_test)
            y_pred_labels = self.label_encoder.inverse_transform(y_pred)

            
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)

            results[name] = {
                'model': best_model,
                'best_params': grid_search.best_params_,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
              
                'classification_report': classification_report(y_test_labels, y_pred_labels),
               
                'confusion_matrix': confusion_matrix(y_test_labels, y_pred_labels, labels=self.label_encoder.classes_)
            }

            print(
                f"{name} - Accuracy: {accuracy:.4f} (+/- {cv_scores.std() * 2:.4f})")

          
            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_model = best_model
                self.best_model_name = name

        self.models = results
        return results

    def evaluate_model(self, model_name):
        """Evalúa un modelo específico en detalle"""
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} no encontrado")

        model_data = self.models[model_name]
        print(f"\n=== Evaluación detallada: {model_name} ===")
        print(f"Mejores parámetros: {model_data['best_params']}")
        print(f"Accuracy: {model_data['accuracy']:.4f}")
        print(
            f"CV Score: {model_data['cv_mean']:.4f} (+/- {model_data['cv_std']:.4f})")
        print("\nReporte de clasificación:")
        print(model_data['classification_report'])
        print("\nMatriz de confusión:")
        print(model_data['confusion_matrix'])

    def save_best_model(self, filepath):
        """Guarda el mejor modelo"""
        if self.best_model is None:
            raise ValueError("No hay modelo entrenado para guardar")

        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'accuracy': self.best_score,
            'feature_columns': self.feature_columns,
            'label_encoder': self.label_encoder 
        }

        joblib.dump(model_data, filepath)
        print(f"Mejor modelo ({self.best_model_name}) guardado en: {filepath}")

    def load_model(self, filepath):
        """Carga un modelo guardado"""
        model_data = joblib.load(filepath)
        self.best_model = model_data['model']
        self.best_model_name = model_data['model_name']
        self.best_score = model_data['accuracy']
        
        self.label_encoder = model_data['label_encoder']
        return model_data

    def predict(self, X):
        """Hace predicciones con el mejor modelo"""
        if self.best_model is None:
            raise ValueError("No hay modelo entrenado")

       
        y_pred_encoded = self.best_model.predict(X)
        
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        return y_pred
