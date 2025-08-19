import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

class CreditScoringModel:
    def __init__(self):
        self.model = DecisionTreeClassifier()
        self.features = ['debt_to_equity', 'current_ratio', 'profit_margin'] # Add more features

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_data):
        return self.model.predict(X_data)[0]

    def get_feature_importance(self):
        """Returns feature importance from the trained model."""
        return dict(zip(self.features, self.model.feature_importances_))

# Note: You'd need to create a training dataset for this.
# A simple approach would be to label historical data based on past credit rating changes.