import os
import sys
import numpy as np
import pandas as pd
from joblib import dump

from src.constants import *
from src.logger import logging
from src.utils.utils import dataset
from src.exception import CustomException
from sklearn.preprocessing import MinMaxScaler
from src.entity.config_entity import DataPreprocessingConfig
from src.entity.artifacts_entity import (
    DataIngestionArtifacts,
    DataPreprocessingArtifacts,
)


class DataPreprocessing:
    def __init__(
        self,
        data_ingestion_artifacts: DataIngestionArtifacts,
        data_preprocessing_config: DataPreprocessingConfig,
    ):
        self.data_ingestion_artifacts = data_ingestion_artifacts
        self.data_preprocessing_config = data_preprocessing_config

    def data_preprocessing(self):
        try:
            logging.info("Data preprocessing started..")
            df = pd.read_csv(self.data_ingestion_artifacts.data_file_path)
            df.set_index("Date", inplace=True)
            df_close = df[[OHLC_CHOICE]]

            # Converting the dataframe to numpy array
            df_arr = np.array(df_close).reshape(-1, 1)

            # Spliting the dataset into training set and test set
            training_size = int(len(df_arr) * TRAINING_SIZE)
            train_data = df_arr[:training_size]
            test_data = df_arr[training_size:]

            # Feature engineering
            scaler = MinMaxScaler(feature_range=(0, 1))
            train_data = scaler.fit_transform(train_data)
            test_data = scaler.transform(test_data)
            logging.info("Data preprocessed successfully...")
            return scaler, train_data, test_data

        except Exception as e:
            raise CustomException(e, sys)

    def create_dataset(self, train_data, test_data):
        try:
            X_train, y_train = dataset(train_data, TIME_STEPS)
            X_test, y_test = dataset(test_data, TIME_STEPS)

            # reshape input which is required for LSTM
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            return X_train, y_train, X_test, y_test
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_preprocessing(self):
        try:
            logging.info("Initiate data preprocessing...")
            scaler, train_data, test_data = self.data_preprocessing()
            X_train, y_train, X_test, y_test = self.create_dataset(
                train_data, test_data
            )

            os.makedirs(
                self.data_preprocessing_config.DATA_PREPROCESS_ARTIFACTS_DIR,
                exist_ok=True,
            )
            os.makedirs(self.data_preprocessing_config.TRAIN_DIR, exist_ok=True)
            os.makedirs(self.data_preprocessing_config.TEST_DIR, exist_ok=True)

            np.save(self.data_preprocessing_config.X_TRAIN_FILE_PATH, X_train)
            np.save(self.data_preprocessing_config.Y_TRAIN_FILE_PATH, y_train)

            np.save(self.data_preprocessing_config.X_TEST_FILE_PATH, X_test)
            np.save(self.data_preprocessing_config.Y_TEST_FILE_PATH, y_test)

            dump(scaler, self.data_preprocessing_config.SCALER_FILE_PATH)

            data_preprocessing_artifacts = DataPreprocessingArtifacts(
                self.data_preprocessing_config.X_TRAIN_FILE_PATH,
                self.data_preprocessing_config.Y_TRAIN_FILE_PATH,
                self.data_preprocessing_config.X_TEST_FILE_PATH,
                self.data_preprocessing_config.Y_TEST_FILE_PATH,
                self.data_preprocessing_config.SCALER_FILE_PATH,
            )
            logging.info("Data preprocessing completed successfully..")
            return data_preprocessing_artifacts

        except Exception as e:
            raise CustomException(e, sys)
