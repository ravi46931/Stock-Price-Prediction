import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.pipeline.train_pipeline import TrainPipeline
from src.pipeline.visual_pipeline import VisualPipeline
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.entity.artifacts_entity import (
    DataIngestionArtifacts,
    DataPreprocessingArtifacts,
    ModelTrainerArtifacts,
    ModelEvaluationArtifacts,
)


class MainPipeline:
    def __init__(self):
        pass

    def start_train_pipeline(self):
        try:
            train_pipeline = TrainPipeline()
            train_pipeline.run_train_pipeline()

            data_ingestion_artifacts = train_pipeline.data_ingestion_artifacts
            data_preprocessing_artifacts = train_pipeline.data_preprocessing_artifacts
            model_trainer_artifacts = train_pipeline.model_trainer_artifacts
            model_evaluation_artifacts = train_pipeline.model_evaluation_artifacts

            return (
                data_ingestion_artifacts,
                data_preprocessing_artifacts,
                model_trainer_artifacts,
                model_evaluation_artifacts,
            )

        except Exception as e:
            raise CustomException(e, sys)

    def start_visual_pipeline(
        self,
        data_ingestion_artifacts: DataIngestionArtifacts,
        data_preprocessing_artifacts: DataPreprocessingArtifacts,
        model_trainer_artifacts: ModelTrainerArtifacts,
        model_evaluation_artifacts: ModelEvaluationArtifacts,
    ):
        try:
            visual_pipeline = VisualPipeline(
                data_ingestion_artifacts,
                data_preprocessing_artifacts,
                model_trainer_artifacts,
                model_evaluation_artifacts,
            )
            visual_pipeline.run_visual_pipeline()
        except Exception as e:
            raise CustomException(e, sys)

    def start_prediction_pipeline(
        self,
        data_preprocessing_artifacts: DataPreprocessingArtifacts,
        model_trainer_artifacts: ModelTrainerArtifacts,
    ):
        try:
            prediction_pipeline = PredictionPipeline(
                data_preprocessing_artifacts, model_trainer_artifacts
            )
            prediction_pipeline.run_prediction_pipeline()
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            logging.info("Start running main pipeline...")

            (
                data_ingestion_artifacts,
                data_preprocessing_artifacts,
                model_trainer_artifacts,
                model_evaluation_artifacts,
            ) = self.start_train_pipeline()
            self.start_visual_pipeline(
                data_ingestion_artifacts,
                data_preprocessing_artifacts,
                model_trainer_artifacts,
                model_evaluation_artifacts,
            )
            self.start_prediction_pipeline(
                data_preprocessing_artifacts, model_trainer_artifacts
            )

            logging.info("Main pipeline run successfully...")

        except Exception as e:
            raise CustomException(e, sys)
