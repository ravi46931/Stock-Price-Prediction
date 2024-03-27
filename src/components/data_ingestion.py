import os
import sys
import yfinance as yf
from pandas_datareader import data as pdr

from src.constants import *
from src.logger import logging
from src.utils.utils import time
from src.exception import CustomException
from src.entity.artifacts_entity import DataIngestionArtifacts
from src.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    def data_ingestion(self):
        try:
            logging.info("Fetching the data from the source..")
            yf.pdr_override()
            current_time, ten_years_ago = time()
            df = pdr.DataReader(STOCK, start=ten_years_ago, end=current_time)
            logging.info("Data fetched successfully..")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        try:
            logging.info("Initiating data ingestion..")
            df = self.data_ingestion()
            os.makedirs(
                self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, exist_ok=True
            )
            df.to_csv(self.data_ingestion_config.RAW_FILE_PATH)

            data_ingestion_artifacts = DataIngestionArtifacts(
                self.data_ingestion_config.RAW_FILE_PATH
            )
            logging.info("Data Ingestion completed")
            return data_ingestion_artifacts
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion(DataIngestionConfig())
    obj.initiate_data_ingestion()
