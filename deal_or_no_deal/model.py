import random

import joblib
from sklearn.linear_model import Lasso
import xgboost as xgb


class BankerModel():
    def __init__(self):
        self.preference_multiplier = 0.075

    def fit(self, X, y):
        """TODO."""
        print('Fitting linear model...')
        self._train_linear_model(X, y)
        print('Fitting XGBoost model...')
        self._train_xgboost(X, y)

    def _train_linear_model(self, X, y):
        """TODO."""
        self.linear_model = Lasso(alpha=0.1)
        self.linear_model.fit(X, y)

    def _train_xgboost(self, X, y):
        """TODO."""
        params = {
            'objective': 'reg:squarederror',
            'booster': 'gbtree',
            'verbosity': 1,
            'validate_parameters': True,
            'eta': 0.3,
            'gamma': 0.0,
            'max_depth': 10,
            'min_child_weight': 15.0,
            'max_delta_step': 0.0,
            'subsample': 0.75,
            'sampling_method': 'uniform',
            'colsample_bytree': 1.0,
            'lambda': 0.001,
            'alpha': 0.1,
            'tree_method': 'exact',
            'eval_metric': 'rmse',
            'n_estimators': 1000,
        }

        self.xgb_model = xgb.XGBRegressor(**params)
        self.xgb_model.fit(X, y)

    def predict(self, X, we_like_contestant=None):
        """TODO."""
        preference_factor = 1 + (random.random() * self.preference_multiplier)
        weight_factor = random.random()

        linear_model_prediction = self.linear_model.predict(X)
        xgb_model_prediction = self.xgb_model.predict(X)

        prediction = (
            weight_factor * linear_model_prediction + (1 - weight_factor) * xgb_model_prediction
        )

        if we_like_contestant is True:
            prediction += (prediction * preference_factor)
        elif we_like_contestant is False:
            prediction -= (prediction * preference_factor)

        return prediction

    def save(self, filename):
        """TODO."""
        joblib.dump(self, filename)
