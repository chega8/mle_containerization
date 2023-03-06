"""Logreg model"""

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV


class BaseModel:
    """Base model"""

    def __init__(self) -> None:
        self.params_to_log = {}
        self.sklearn_model = None
        self.params = None

    def train(self, features, target):
        """Train grid search model"""

        gs = GridSearchCV(
            self.sklearn_model, [self.params], cv=3, n_jobs=-1, scoring="roc_auc"
        )
        gs.fit(features, target)
        grid_search_report = gs.cv_results_
        self.parse_params(grid_search_report)
        return gs.best_estimator_

    def parse_params(self, param_grid):
        """Parse grid search results"""
        self.params_to_log["params"] = param_grid["params"]
        self.params_to_log["scores"] = param_grid["mean_test_score"]


class Logreg(BaseModel):
    """Logreg model"""

    def __init__(self) -> None:
        super().__init__()

        self.sklearn_model = LogisticRegression()
        self.params = {"C": [10**-1, 10**0, 10**1]}


class SVM(BaseModel):
    """SVM model"""

    def __init__(self) -> None:
        super().__init__()

        self.sklearn_model = SVC(probability=True)
        self.params = {"C": [10**-1, 10**0, 10**1]}


class GB(BaseModel):
    """Gradient boosting model"""

    def __init__(self) -> None:
        super().__init__()

        self.sklearn_model = GradientBoostingClassifier()
        self.params = {"n_estimators": [10, 50, 100], "max_depth": [5, 10]}
