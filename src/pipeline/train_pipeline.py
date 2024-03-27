import sys

from src.constants import *
from src.logger import logging
from src.exception import CustomException
from src.components.model_trainer import ModelTrainer
from src.components.data_ingestion import DataIngestion
from src.components.model_evaluation import ModelEvaluation
from src.components.data_preprocessing import DataPreprocessing
from src.entity.artifacts_entity import (
    DataIngestionArtifacts,
    DataPreprocessingArtifacts,
    ModelTrainerArtifacts,
)
from src.entity.config_entity import (
    DataIngestionConfig,
    DataPreprocessingConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_preprocessing_config = DataPreprocessingConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()

        self.data_ingestion_artifacts = None
        self.data_preprocessing_artifacts = None
        self.model_trainer_artifacts = None
        self.model_evaluation_artifacts = None

    def start_data_ingestion(self):
        try:
            logging.info(
                "Entered the start_data_ingestion method of TrainPipeline class"
            )
            data_ingestion = DataIngestion(self.data_ingestion_config)
            data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()

            logging.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )

            return data_ingestion_artifacts

        except Exception as e:
            raise CustomException(e, sys)

    def start_data_preprocessing(
        self, data_ingestion_artifacts: DataIngestionArtifacts
    ):
        try:
            logging.info(
                "Entered the start_data_preprocesing method of TrainPipeline class"
            )
            data_preprocessing = DataPreprocessing(
                data_ingestion_artifacts, self.data_preprocessing_config
            )
            data_preprocessing_artifacts = (
                data_preprocessing.initiate_data_preprocessing()
            )
            logging.info(
                "Exited the start_data_preprocesing method of TrainPipeline class"
            )

            return data_preprocessing_artifacts
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_trainer(
        self, data_preprocessing_artifacts: DataPreprocessingArtifacts
    ):
        try:
            logging.info(
                "Entered the start_model_trainer method of TrainPipeline class"
            )
            model_trainer = ModelTrainer(
                data_preprocessing_artifacts, self.model_trainer_config
            )
            model_trainer_artifacts = model_trainer.initiate_model_training()
            logging.info("Exited the start_model_trainer method of TrainPipeline class")

            return model_trainer_artifacts
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_evaluation(
        self,
        data_preprocessing_artifacts: DataPreprocessingArtifacts,
        model_trainer_artifacts: ModelTrainerArtifacts,
    ):
        try:
            logging.info(
                "Entered the start_data_evaluation method of TrainPipeline class"
            )
            model_evaluation = ModelEvaluation(
                data_preprocessing_artifacts,
                model_trainer_artifacts,
                self.model_evaluation_config,
            )
            model_evaluation_artifacts = model_evaluation.initiate_model_evaluation()
            logging.info(
                "Exited the start_data_evaluation method of TrainPipeline class"
            )

            return model_evaluation_artifacts
        except Exception as e:
            raise CustomException(e, sys)

    def run_train_pipeline(self):
        try:
            logging.info("Start running train pipeline...")

            data_ingestion_artifacts = self.start_data_ingestion()
            data_preprocessing_artifacts = self.start_data_preprocessing(
                data_ingestion_artifacts
            )
            model_trainer_artifacts = self.start_model_trainer(
                data_preprocessing_artifacts
            )
            model_evaluation_artifacts = self.start_model_evaluation(
                data_preprocessing_artifacts, model_trainer_artifacts
            )

            self.data_ingestion_artifacts = data_ingestion_artifacts
            self.data_preprocessing_artifacts = data_preprocessing_artifacts
            self.model_trainer_artifacts = model_trainer_artifacts
            self.model_evaluation_artifacts = model_evaluation_artifacts

            logging.info("Train pipeline run successfully...")

        except Exception as e:
            raise CustomException(e, sys)


# if __name__=="__main__":
#     obj = DataIngestion(DataIngestionConfig())
#     data_ingestion_artifacts = obj.initiate_data_ingestion()

#     df = pd.read_csv(data_ingestion_artifacts.data_file_path)
#     save_ohlc(df)

#     obj2 = DataPreprocessing(data_ingestion_artifacts, DataPreprocessingConfig())
#     data_preprocessing_artifacts = obj2.initiate_data_preprocessing()

#     obj3 = ModelTrainer(data_preprocessing_artifacts, ModelTrainerConfig())
#     model_trainer_artifacts = obj3.initiate_model_training()

#     obj4 = ModelEvaluation(data_preprocessing_artifacts, model_trainer_artifacts, ModelEvaluationConfig())
#     model_evaluation_artifacts = obj4.initiate_model_evaluation()

#     model_history_path = model_trainer_artifacts.model_history_file_path
#     # Load the saved history
#     import pickle
#     with open(model_history_path, 'rb') as file:
#         model_history = pickle.load(file)
#     plot_mae_loss(model_history)

#     # Training prediction plot
#     y_train = np.load(data_preprocessing_artifacts.y_train_file_path)
#     train_pred = np.load(model_evaluation_artifacts.train_pred_file_path)
#     scaler = load(data_preprocessing_artifacts.scaler_file_path)
#     plot_pred_actual(y_train, train_pred, scaler, type='training', image_name=PRED_ACT_TRAIN)

#     # Training prediction plot
#     y_test = np.load(data_preprocessing_artifacts.y_test_file_path)
#     test_pred = np.load(model_evaluation_artifacts.test_pred_file_path)
#     scaler = load(data_preprocessing_artifacts.scaler_file_path)
#     plot_pred_actual(y_test, test_pred, scaler, type='test', image_name=PRED_ACT_TEST)

#     custom_plot(y_train, train_pred, scaler, image_name=CUSTOM_PLOT, days=RECENT_DAYS)

#     predpipeline = PredictionPipeline(data_preprocessing_artifacts,model_trainer_artifacts)
#     predpipeline.initiate_prediction()
