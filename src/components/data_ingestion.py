import logging
import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # type: ignore
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import datatransformationconfig

@dataclass
class dataingestionconfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"raw.csv")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = dataingestionconfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion initiated")
        try:        
            df = pd.read_csv('notebooks/data/stud.csv')
            logging.info("Raw data loaded successfully")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info("Raw data saved successfully and train/test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train and Test data saved successfully")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
            
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    train_data, tets_data = DataIngestion().initiate_data_ingestion()
    data_transformer = DataTransformation()
    data_transformer.initiate_data_transformation(train_data, tets_data)