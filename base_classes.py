from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier



class LogisticRegressionModel:
    def __init__(self, numerical_features, categorical_features, time_features):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.time_features = time_features
        self.model = None
        self.param_grid = {
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__solver': ['liblinear', 'saga']
        }

    def create_pipeline(self, categorical_encoder=None):
        if categorical_encoder is None:
            categorical_encoder = OneHotEncoder(
                drop='first', 
                sparse_output=False, 
                handle_unknown='ignore',
                min_frequency=0.001
            )
            
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_features + self.time_features),
                ('cat', categorical_encoder, self.categorical_features)
            ])
            
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
        ])
        return self
    
    def train(self, X_train, y_train, cv=3):
        self.model = GridSearchCV(
            self.pipeline,
            self.param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            error_score='raise'
        )
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]
    
    def get_best_params(self):
        return self.model.best_params_
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return roc_auc_score(y_test, y_pred)


class NaiveBayesModel:
    def __init__(self, numerical_features, categorical_features, time_features, nb_type='gaussian'):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.time_features = time_features
        self.model = None
        self.nb_type = nb_type

    def create_pipeline(self, categorical_encoder=None):
        if categorical_encoder is None:
            categorical_encoder = OneHotEncoder(
                drop='first',
                sparse_output=False,
                handle_unknown='ignore',
                min_frequency=0.001
            )
        
        # Выбираем тип наивного Байеса
        if self.nb_type == 'gaussian':
            classifier = GaussianNB()
        elif self.nb_type == 'bernoulli':
            classifier = BernoulliNB()
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', self.numerical_features + self.time_features),
                ('cat', categorical_encoder, self.categorical_features)
            ])
        
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])
        return self
    
    def train(self, X_train, y_train):
        self.model = self.pipeline
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return roc_auc_score(y_test, y_pred)


class RandomForestModel:
    def __init__(self, numerical_features, categorical_features, time_features):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.time_features = time_features
        self.model = None
        self.param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 10, 15],
            'classifier__min_samples_split': [2, 5, 7],
            'classifier__class_weight': ['balanced']
        }

    def create_pipeline(self, categorical_encoder=None):
        if categorical_encoder is None:
            categorical_encoder = OneHotEncoder(
                drop='first',
                sparse_output=False,
                handle_unknown='ignore',
                min_frequency=0.001
            )
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_features + self.time_features),
                ('cat', categorical_encoder, self.categorical_features)
            ])
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier())
        ])
        return self


    def train(self, X_train, y_train, cv=3):
        self.model = GridSearchCV(
            self.pipeline,
            self.param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            error_score='raise'
        )
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get_best_params(self):
        return self.model.best_params_

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return roc_auc_score(y_test, y_pred)


class XGBoostModel:
    def __init__(self, numerical_features, categorical_features, time_features):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.time_features = time_features
        self.model = None
        self.param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.1, 1, 10]
        }

    def create_pipeline(self, categorical_encoder=None):
        if categorical_encoder is None:
            categorical_encoder = OneHotEncoder(
                drop='first',
                sparse_output=False,
                handle_unknown='ignore',
                min_frequency=0.001
            )
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_features + self.time_features),
                ('cat', categorical_encoder, self.categorical_features)
            ])
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', xgb.XGBClassifier(eval_metric='logloss'))
        ])
        return self

    def train(self, X_train, y_train, cv=3):
        self.model = GridSearchCV(
            self.pipeline,
            self.param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            error_score='raise'
        )
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get_best_params(self):
        return self.model.best_params_

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return roc_auc_score(y_test, y_pred)


class DecisionTreeModel:
    def __init__(self, numerical_features, categorical_features, time_features):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.time_features = time_features
        self.model = None
        self.param_grid = {
            'classifier__max_depth': [None, 5, 10, 15],
            'classifier__min_samples_split': [2, 5, 7],
            'classifier__min_samples_leaf': [1, 2, 4, 6, 10],
            'classifier__class_weight': ['balanced', None]
        }

    def create_pipeline(self, categorical_encoder=None):
        if categorical_encoder is None:
            categorical_encoder = OneHotEncoder(
                drop='first',
                sparse_output=False,
                handle_unknown='ignore',
                min_frequency=0.001
            )
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_features + self.time_features),
                ('cat', categorical_encoder, self.categorical_features)
            ])
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier())
        ])
        return self

    def train(self, X_train, y_train, cv=3):
        self.model = GridSearchCV(
            self.pipeline,
            self.param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            error_score='raise'
        )

        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get_best_params(self):
        return self.model.best_params_


    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return roc_auc_score(y_test, y_pred)
    
class AdaBoostModel:
    def __init__(self, numerical_features, categorical_features, time_features):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.time_features = time_features
        self.model = None
        self.param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 1.0],
            'classifier__estimator__max_depth': [1, 2, 3]
        }

    def create_pipeline(self, categorical_encoder=None):
        if categorical_encoder is None:
            categorical_encoder = OneHotEncoder(
                drop='first',
                sparse_output=False,
                handle_unknown='ignore',
                min_frequency=0.001
            )
        
        base_estimator = DecisionTreeClassifier(max_depth=3)
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_features + self.time_features),
                ('cat', categorical_encoder, self.categorical_features)
            ])
        
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', AdaBoostClassifier(
                estimator=base_estimator,
                random_state=42
            ))
        ])
        return self

    def train(self, X_train, y_train, cv=3):
        self.model = GridSearchCV(
            self.pipeline,
            self.param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            error_score='raise'
        )
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get_best_params(self):
        return self.model.best_params_

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return roc_auc_score(y_test, y_pred)
