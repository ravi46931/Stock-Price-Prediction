import os
from dataclasses import dataclass
from src.constants import *


@dataclass
class DataIngestionConfig:
    DATA_INGESTION_ARTIFACTS_DIR: str = os.path.join(
        ARTIFACTS, DATA_INGESTION_ARTIFACTS_DIR
    )
    RAW_FILE_PATH: str = os.path.join(DATA_INGESTION_ARTIFACTS_DIR, RAW_FILE_NAME)


@dataclass
class DataPreprocessingConfig:
    DATA_PREPROCESS_ARTIFACTS_DIR: str = os.path.join(
        ARTIFACTS, DATA_PREPROCESS_ARTIFACTS_DIR
    )

    TRAIN_DIR: str = os.path.join(DATA_PREPROCESS_ARTIFACTS_DIR, TRAIN_DIR)
    TEST_DIR: str = os.path.join(DATA_PREPROCESS_ARTIFACTS_DIR, TEST_DIR)
    SCALER_FILE_PATH: str = os.path.join(
        DATA_PREPROCESS_ARTIFACTS_DIR, SCALER_FILE_NAME
    )

    X_TRAIN_FILE_PATH: str = os.path.join(TRAIN_DIR, X_TRAIN_FILE_NAME)
    Y_TRAIN_FILE_PATH: str = os.path.join(TRAIN_DIR, Y_TRAIN_FILE_NAME)

    X_TEST_FILE_PATH: str = os.path.join(TEST_DIR, X_TEST_FILE_NAME)
    Y_TEST_FILE_PATH: str = os.path.join(TEST_DIR, Y_TEST_FILE_NAME)


@dataclass
class ModelTrainerConfig:
    MODEL_TRAINER_ARTIFACTS_DIR: str = os.path.join(
        ARTIFACTS, MODEL_TRAINER_ARTIFACTS_DIR
    )
    MODEL_FILE_PATH: str = os.path.join(MODEL_TRAINER_ARTIFACTS_DIR, MODEL_FILE)
    MODEL_HISTORY_FILE_PATH: str = os.path.join(
        MODEL_TRAINER_ARTIFACTS_DIR, MODEL_HISTORY_FILE
    )


@dataclass
class ModelEvaluationConfig:
    MODEL_EVALUATION_ARTIFACTS_DIR: str = os.path.join(
        ARTIFACTS, MODEL_EVALUATION_ARTIFACTS_DIR
    )
    METRICS_FILE_PATH: str = os.path.join(
        MODEL_EVALUATION_ARTIFACTS_DIR, METRICS_FILE_NAME
    )
    TRAIN_PRED_FILE_PATH: str = os.path.join(
        MODEL_EVALUATION_ARTIFACTS_DIR, TRAIN_PRED_FILE_NAME
    )
    TEST_PRED_FILE_PATH: str = os.path.join(
        MODEL_EVALUATION_ARTIFACTS_DIR, TEST_PRED_FILE_NAME
    )
