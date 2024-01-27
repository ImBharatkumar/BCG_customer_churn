import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.components import save_to_csv

class DataTransformation():
    def __init__(self,path):
        self.data = path

    def load_data(self):
        
        try:
            logging.info("Loading data from file: {file}".format(file=self.clean_data_file))
            self.data = pd.read_csv(self.data)
            logging.info("Data loaded successfully")
            return self.data, self.price_df
        except Exception as e:
            raise CustomException("Unable to load data from file: {file}".format(file=self.clean_data_file), e)

    def transform_data(self):
        """
        Method to perform data transformation on the loaded data
        """
        try:
            logging.info("Starting data transformation")
            # Apply log10 transformation
            skewed=["cons_12m","cons_gas_12m","cons_last_month","forecast_cons_12m"
                    ,"forecast_cons_year","forecast_meter_rent_12m""imp_cons"]
            
            for feature in skewed:
                self.data[feature]=np.log10(self.data[feature] + 1)


            logging.info("Data transformation completed successfully")


            return self.data
        
        except Exception as e:
            raise CustomException("Unable to transform data", e)