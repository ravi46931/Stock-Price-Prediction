import os
import sys
from pathlib import Path


list_of_files = [
    "src/__init__.py",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_preprocessing.py",
    "src/components/model_trainer.py",
    "src/components/model_evaluation.py",
    "src/entity/__init__.py", 
    "src/entity/config_entity.py", 
    "src/entity/artifacts_entity.py", 
    "src/constants/__init__.py", 
    "src/logger/__init__.py", 
    "src/exception/__init__.py", 
    "src/utils/__init__.py", 
    "src/utils/utils.py", 
    "src/pipeline/train_pipeline.py",
    "src/pipeline/prediction_pipeline.py",
    "src/pipeline/visual_pipeline.py",
    "src/pipeline/main_pipeline.py",
    "src/pipeline/__init__.py",
    "src/ml/__init__.py", 
    "src/ml/model.py", 
    "src/ml/metrics.py", 
    "src/ml/standardization.py", 
    "src/visualization/ohlc_plot.py", 
    "src/visualization/mae_loss.py", 
    "src/visualization/pred_act.py", 
    "src/visualization/forecast.py", 

    "experiments/.gitkeep", 
    "static/css/style.css",
    "static/css/mediaqueries.css",
    "templates/index.html", 
    "templates/base.html", 
    "templates/layout.html",
    "requirements.txt",
    "README.md",
    "setup.py",
    "app.py", 
    "main.py", 
    "demo.py"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (
        os.path.getsize(filepath) == 0
    ):  # This line will be responsible for ERROR
        with open(filepath, "w") as f:
            pass
    else:
        pass
