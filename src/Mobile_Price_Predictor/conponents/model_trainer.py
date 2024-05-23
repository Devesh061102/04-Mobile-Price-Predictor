import os
from Mobile_Price_Predictor.logging import logger
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import pandas as pd
from Mobile_Price_Predictor.entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

        
    def train(self):
            # Load data (assumes data is in CSV format with features and a target column)
            data = pd.read_csv(self.config.train_data_path)
            X = data.drop(columns=[self.config.target_column])
            y = data[self.config.target_column]

            # Split data into train and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            # Standardize features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

            # Define the model with hyperparameters
            model_svc = SVC(C=self.config.C, gamma=self.config.gamma, kernel=self.config.kernel)


            # Train the model
            model_svc.fit(X_train, y_train)

            # Evaluate the model
            y_pred = model_svc.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            print(f"Validation Accuracy: {accuracy:.4f}")

            # Save the model and the scaler
            model_path = os.path.join(self.config.trained_model_path, "svc_model.pkl")
            joblib.dump(model_svc, model_path)
            logger.info("Model Trained and Saved Successfully")