# common constants
ARTIFACTS = "artifacts"

# Data Ingestion Constants
DATA_INGESTION_ARTIFACTS_DIR = "DataIngestionArtifacts"
RAW_FILE_NAME = "data.csv"
"""
Google: GOOGL
Amazon.com Inc.: AMZN
Microsoft Corporation: MSFT
Facebook (Meta Platforms, Inc.): FB
Tesla, Inc.: TSLA
Apple: AAPL
"""
STOCK = "GOOGL"
DAYS = 10 * 365  # 10 years data it will fetch from current date
##############################################

STATIC_DIR = "static"
IMAGE_DIR = "images"


# OHLC Plot
OHLC_FIGSIZE = (8, 6)
OHLC_IMAGE_NAME = "ohlc.png"
##############################################

# Data Preprocessing Constants
DATA_PREPROCESS_ARTIFACTS_DIR = "DataPreprocessArtifacts"
TRAIN_DIR = "train"
TEST_DIR = "test"
X_TRAIN_FILE_NAME = "x_train.npy"
Y_TRAIN_FILE_NAME = "y_train.npy"
X_TEST_FILE_NAME = "x_test.npy"
Y_TEST_FILE_NAME = "y_test.npy"
SCALER_FILE_NAME = "scaler.joblib"
OHLC_CHOICE = "Close"
TRAINING_SIZE = 0.75  # (0, 1)
TIME_STEPS = 100  # Number of days

# Model Tarining constants
MODEL_TRAINER_ARTIFACTS_DIR = "ModelTrainerArtifacts"
MODEL_FILE = "model.h5"
MODEL_HISTORY_FILE = "model_history.pkl"
FILTERS = 64
KERNEL_SIZE = 3
STRIDES = 1
ACTIVATION = "relu"
PADDING = "causal"
LEARNING_RATE = 0.001
METRICS = "mae"
EPOCHS = 100
BATCH_SIZE = 64
VERBOSE = 1

# Model Evaluation Constants
MODEL_EVALUATION_ARTIFACTS_DIR = "ModelEvaluationArtifacts"
METRICS_FILE_NAME = "metrics.json"
TRAIN_PRED_FILE_NAME = "train_pred.npy"
TEST_PRED_FILE_NAME = "test_pred.npy"


## MAE and LOSS pics
MAE_LOSS = "mae_loss.png"
MAE_LOSS_ZOOM = "mae_loss_zoom.png"
##
PRED_ACT_TRAIN = "pred_act_train.png"
PRED_ACT_TEST = "pred_act_test.png"
##
CUSTOM_PLOT = "custom_plot.png"
RECENT_DAYS = 150

# Prediction Pipeline Constants
PREDICTION_DAYS = 150
FORECAST_PLOT = "forecast.png"
