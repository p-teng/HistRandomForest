from sklearn.ensemble import BaggingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
'''
Histogram Random Forest Classifier created using combination of HistGradientBoostingClassifier and BaggingClassifier
'''
class HistRandomForestClassifier():
    def __init__(self, loss='auto', 
            max_leaf_nodes=31,
            max_depth=None, 
            min_samples_leaf=20,
            l2_regularization=0, 
            max_bins=255, 
            n_estimators=20,
            max_samples=1.0,
            bootstrap=True,
            bootstrap_features=False,
            oob_score=False,
            categorical_features=None,
            monotonic_cst=None,
            warm_start=False,
            n_jobs=None,
            early_stopping='auto',
            scoring='loss',
            validation_fraction=0.1,
            n_iter_no_change=10,
            tol=1e-7,
            verbose=0,
            random_state=None):
        self.loss = loss
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_bins = max_bins
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score 
        self.categorical_features = categorical_features
        self.monotonic_cst = monotonic_cst
        self.warm_start = warm_start
        self.n_jobs = n_jobs 
        self.early_stopping = early_stopping
        self.scoring = scoring
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        
        self.tree = HistGradientBoostingClassifier(loss=loss, learning_rate=1, max_iter=1, 
                                                max_leaf_nodes=max_leaf_nodes, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                                l2_regularization=l2_regularization, max_bins=max_bins, categorical_features=categorical_features,
                                                monotonic_cst=monotonic_cst, early_stopping=early_stopping, 
                                                scoring=scoring, validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change,
                                                tol=tol, verbose=verbose, random_state=random_state)
        self.HistRF = BaggingClassifier(base_estimator=self.tree, n_estimators=n_estimators, 
                                        bootstrap=bootstrap, bootstrap_features=bootstrap_features, oob_score=oob_score,
                                        warm_start=warm_start, n_jobs=n_jobs, random_state=random_state, verbose=verbose)
        
    def decision_function(self, X):
        return self.HistRF.decision_function(X)
    
    def fit(self, X, y, sample_weight=None):
        self.HistRF.fit(X, y, sample_weight=sample_weight)
        return self.HistRF
    
    def get_params(self, deep=True):
        return self.HistRF.get_params(deep=deep)
    
    def predict(self, X):
        return self.HistRF.predict(X)

    def predict_log_proba(self, X):
        return self.HistRF.predict_log_proba(X)
    
    def predict_proba(self, X):
        return self.HistRF.predict_proba(X)
    
    def score(self, X, y, sample_weight=None):
        return self.HistRF.score(X, y, sample_weight=sample_weight)
    
    def set_params(self, **params):
        return self.HistRF.set_params(**params)
