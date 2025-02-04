#dataengineering, data cleaning
import logging
import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer #for pipelining
from sklearn.pipeline import Pipeline   #for pipelining
from sklearn.impute import SimpleImputer    #for pipelining
from sklearn.preprocessing import StandardScaler, OneHotEncoder    #for pipelining

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class datatransformationconfig:
    preprocessordata_path: str=os.path.join('artifacts',"preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = datatransformationconfig()
        
    def get_data_transformer(self):
        ''' handling valeus of all kind of data
        
        Returns:...'''
        logging.info("Data Transformation initiated")
        try:
            df = pd.read_csv('notebooks/data/stud.csv')
            logging.info("Raw data loaded successfully")
            
            #preprocessing
            numeric_features = ['writing_score', 'reading_score']
            categorical_columns = [
                'gender',
                "race_ethinicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),           # handling missing values
                ('scaler', StandardScaler())]
                                           )
            
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')), # handling missing values
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ("scaler",StandardScaler())]                                               
                                               )
            logging.info("numerical_features standard scaling done")
            logging.info("categorical_columns encoding done")
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', numeric_transformer, numeric_features),
                    ('cat_pipeline', categorical_transformer, categorical_columns)
                ]
            )
            
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        '''initiate data transformation
        
        Args:
            train_path: path to train data
            test_path: path to test data
        
        Returns:...'''
        logging.info("Data Transformation initiated")
        try:
            logging.info("obtaining preprocessor object")
            
            
            target_columns = ["math_score"]
            numeric_features = ['writing_score', 'reading_score']
            
            preprocessor = self.get_data_transformer()
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            
            '''
            difference between fit and fit_transform:
            fit: calculates the parameters (e.g. μ and σ in case of StandardScaler) and saves them as an internal objects state.
            transform: it takes input data and parameters calculated by fit() and return transformed data.
            fit_transform: joins the fit() and transform() method for transformation of dataset.'''
            
            X_train = preprocessor.fit_transform(train_data.drop(target_columns, axis=1))
            X_train_arr =np.c_[X_train, np.array(train_data["math_score"])]
            
            X_test = preprocessor.transform(test_data.drop(target_columns, axis=1))
            X_test_arr = np.c_[X_test,np.array(test_data["math_score"])]
            
            logging.info("Data Transformation completed")
            save_object(
                file_path = self.data_transformation_config.preprocessordata_path,
                obj = preprocessor
            )
            
            return(X_train_arr, X_test_arr, self.data_transformation_config.preprocessordata_path)
        except Exception as e:
            raise CustomException(e,sys)