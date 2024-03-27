from dataclasses import dataclass


@dataclass
class DataIngestionArtifacts:
    data_file_path: str


@dataclass
class DataPreprocessingArtifacts:
    x_train_file_path: str
    y_train_file_path: str
    x_test_file_path: str
    y_test_file_path: str
    scaler_file_path: str


@dataclass
class ModelTrainerArtifacts:
    model_file_path: str
    model_history_file_path: str


@dataclass
class ModelEvaluationArtifacts:
    metrics_file_path: str
    train_pred_file_path: str
    test_pred_file_path: str
