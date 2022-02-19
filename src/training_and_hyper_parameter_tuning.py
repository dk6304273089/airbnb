import pickle
import pandas as pd
import numpy as np
from extract_data import read_params,log
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import json
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import argparse
params={
 "learning_rate": [0.05, 0.10, 0.15, ] ,
 "max_depth": [ 3,  6, 8,  15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma": [ 0.0 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4 , 0.7 ]
}
def get_data(config_path):
    config = read_params(config_path)
    data_path = config["Data"]["Final"]
    df = pd.read_csv(data_path, sep=",")
    return df
class hyper:
    def __init__(self):
        self.file=open("Training_logs/Training_log.txt","a+")
    def tuning(self,config_path):
        try:
            config = read_params(config_path)
            xgboost=config["models"]["xgboost"]
            params_file=config["reports"]["params"]
            scores_file = config["reports"]["scores"]
            standardscaler = config["models"]["sc"]
            df = get_data(config_path)
            X = df.drop("price", axis=1)
            y = df["price"]
            # splitting our data into 70 % for training and 30 % for testing
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            log(self.file, "Stage-5 => successfully splitted data in the ratio of 7:3")
            # normalizing our data
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            log(self.file, "Stage-5 => successfully implemented normalization")
            model = XGBRegressor()
            log(self.file, "Stage-5 => Training started")
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            log(self.file, "Stage-5 => Completed")
            log(self.file, "Stage-5 => Hyper parameter tuning started")
            random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=5, scoring='neg_mean_absolute_error',
                                           n_jobs=-1, cv=5, verbose=3)
            random_search.fit(X, y)
            random_search.best_estimator_.fit(X_train, y_train)
            log(self.file, "Stage-5 => Hyper parameter tuning completed")
            predictions = random_search.best_estimator_.predict(X_test)
            with open(scores_file, "w") as f:
                scores = {
                    "mean_squared_error": mean_squared_error(y_test, predictions),
                    "root_mean_squared_error": sqrt(mean_squared_error(y_test, predictions))
                }
                json.dump(scores, f, indent=4)
            log(self.file, "Stage-5 => Model of the accuracy can found in this file {}".format(scores_file))
            with open(params_file,"w") as f:
                param={"base_score": random_search.best_estimator_.base_score, "booster": random_search.best_estimator_.booster,
                 "colsample_bylevel": random_search.best_estimator_.colsample_bylevel,
                 "colsample_bynode": random_search.best_estimator_.colsample_bynode, "colsample_bytree": random_search.best_estimator_.colsample_bytree,
                 "enable_categorical": random_search.best_estimator_.enable_categorical,
                 "gamma": random_search.best_estimator_.gamma, "gpu_id": random_search.best_estimator_.gpu_id,
                 "importance_type": random_search.best_estimator_.importance_type,
                 "interaction_constraints": random_search.best_estimator_.interaction_constraints,
                 "learning_rate": random_search.best_estimator_.learning_rate, "max_delta_step": random_search.best_estimator_.max_delta_step,
                 "max_depth": random_search.best_estimator_.max_depth, "min_child_weight": random_search.best_estimator_.min_child_weight,
                 "monotone_constraint": random_search.best_estimator_.monotone_constraints, "n_estimators": random_search.best_estimator_.n_estimators,
                 "n_jobs": random_search.best_estimator_.n_jobs,
                 "num_parallel_tree": random_search.best_estimator_.num_parallel_tree, "predictor": random_search.best_estimator_.predictor,
                 "random_state": random_search.best_estimator_.random_state, "reg_alpha": random_search.best_estimator_.reg_alpha,
                 "reg_lambda": random_search.best_estimator_.reg_lambda, "scale_pos_weight": random_search.best_estimator_.scale_pos_weight,
                 "subsample": random_search.best_estimator_.subsample, "tree_method": random_search.best_estimator_.tree_method,
                 "validate_parameters": random_search.best_estimator_.validate_parameters, "verbosity": random_search.best_estimator_.verbosity}
                json.dump(param, f, indent=4)
            log(self.file, "Stage-5 => parameters of the model can found in this file {}".format(params_file))
            pickle.dump(random_search.best_estimator_, open(xgboost, 'wb'))
            pickle.dump(sc, open(standardscaler, 'wb'))
            log(self.file, "Stage-5 => Training and Hyper parameter tuning Done successfully completed and  saved the  model in {}".format(xgboost))
        except ValueError as e:
            log(self.file, "Stage-5 => Error: {}".format(str(e)))
if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config",default="config.yaml")
    parsed_args=args.parse_args()
    hyper().tuning(config_path=parsed_args.config)