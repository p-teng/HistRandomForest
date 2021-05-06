from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
'''
Random Forest Classifier created using combination of HistGradientBoostingClassifier and BaggingClassifier
'''
class BaggedDecisionTreeClassifier():
    def __init__(self, n_estimators=20, bootstrap=True, bootstrap_features=False,
            oob_score=False, max_depth=None, min_samples_leaf=20, warm_start=False,
            n_jobs=None,
            early_stopping='auto',
            verbose=0,
            random_state=None):
        self.tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        self.BagDT = BaggingClassifier(base_estimator=self.tree, n_estimators=n_estimators, 
                                        bootstrap=bootstrap, bootstrap_features=bootstrap_features, oob_score=oob_score,
                                        warm_start=warm_start, n_jobs=n_jobs, random_state=random_state, verbose=verbose)
        
    def decision_function(self, X):
        return self.BagDT.decision_function(X)
    
    def fit(self, X, y, sample_weight=None):
        self.BagDT.fit(X, y, sample_weight=sample_weight)
        return self.BagDT
    
    def get_params(self, deep=True):
        return self.BagDT.get_params(deep=deep)
    
    def predict(self, X):
        return self.BagDT.predict(X)

    def predict_log_proba(self, X):
        return self.BagDT.predict_log_proba(X)
    
    def predict_proba(self, X):
        return self.BagDT.predict_proba(X)
    
    def score(self, X, y, sample_weight=None):
        return self.BagDT.score(X, y, sample_weight=sample_weight)
    
    def set_params(self, **params):
        return self.BagDT.set_params(**params)
