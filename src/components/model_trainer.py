import os
import sys
from dataclasses import dataclass
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    model_path: str=os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    
    def initaiate_model_training(self,train_array, test_array):
        logging.info("Model Training initiated")
        try:
            
            '''
            if train array:[[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]]
                            
            then train_array[:, :-1] will return [[1, 2],
                                                  [4, 5],
                                                  [7, 8]]
                                                  
            and train_array[:, -1] will return [3, 6, 9] 
            '''
            logging.info("Splitting data into train and test")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],)
            logging.info("Train data loaded successfully")
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Linear Regression": LinearRegression(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBRegressor": XGBRegressor(),
                "KNN": KNeighborsRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            }
            
            
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "KNN": {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                "XGBRegressor":{
                    'learning_rate':[0.1,0.01,0.05,0.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            
            model_report: dict = evaluate_models(X_train = X_train, y_train = y_train,X_test=X_test, y_test=y_test, models = models, params = params)
            
            best_model_score = max(sorted(model_report.values()))
            
            ''''
            # model_report example:
            # model_report = {
            # 'model_a': 0.85,
            # 'model_b': 0.90,
            # 'model_c': 0.88
            # }
            # best_model_score = 0.90
            
            # Steps to find the best model:
            # 1. model_report.keys() returns dict_keys(['model_a', 'model_b', 'model_c']).
            # 2. list(model_report.keys()) converts it to ['model_a', 'model_b', 'model_c'].
            # 3. model_report.values() returns dict_values([0.85, 0.90, 0.88]).
            # 4. list(model_report.values()) converts it to [0.85, 0.90, 0.88].
            # 5. list(model_report.values()).index(0.90) finds the index of 0.90, which is 1.
            # 6. list(model_report.keys())[1] accesses the element at index 1 in the list of model names, which is 'model_b'.
            '''
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("No best model found", sys)
            logging.info(f"Best model found overall: {best_model_name}")
            
            save_object(file_path=self.model_trainer_config.model_path, obj=best_model)
            
            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)
            return r2
        except Exception as e:
            raise CustomException(e,sys)