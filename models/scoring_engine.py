import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging

class CreditScoringModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        # We will now set features dynamically in the training script
        self.features = []

    def train(self, X_train, y_train):
        # Ensure the model's features are set before training
        if not self.features:
            self.features = list(X_train.columns)
        self.model.fit(X_train[self.features], y_train)

    def predict(self, X_data):
        return self.model.predict(X_data[self.features])[0]

    def get_feature_importance(self):
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.features, self.model.feature_importances_))
        return None

    # --- ADD/UPDATE THE FOLLOWING METHODS ---
    def save_model(self, file_path):
        """Saves the trained model and features to a file."""
        model_data = {'model': self.model, 'features': self.features}
        joblib.dump(model_data, file_path)
        logging.info(f"Model and features saved to {file_path}")

    @classmethod
    def load_model(cls, file_path):
        """Loads a model and its features from a file."""
        try:
            model_data = joblib.load(file_path)
            model_instance = cls()
            model_instance.model = model_data['model']
            model_instance.features = model_data['features']
            logging.info(f"Model and features loaded from {file_path}")
            return model_instance
        except FileNotFoundError:
            logging.error(f"Model file not found at {file_path}. Please train the model first.")
            return None