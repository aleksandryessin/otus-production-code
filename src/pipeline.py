"""
Main pipeline for the project
"""

import joblib
from loguru import logger
import sys
import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import load_data, split_data
from src.train import train_model, evaluate_model

def main() -> None:
    logger.info("-----------------")
    logger.info("Starting pipeline")
    logger.info(f"Current working directory: {os.getcwd()}")

    logger.info("Loading data")
    df = load_data()

    logger.info("Splitting data")
    train, test = split_data(df)

    logger.info("Training model")
    model = train_model(train)

    logger.info("Saving model")
    try:
        # Обновите путь к правильному местоположению папки models
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, "model_in_docker.joblib")
        logger.info(f"Saving model to: {model_path}")
        joblib.dump(model, model_path)
        if os.path.exists(model_path):
            logger.info("Model file exists")
        else:
            logger.error("Model file does not exist")
    except Exception as e:
        logger.error(f"Error saving model: {e}")

    logger.info("Evaluating model")
    accuracy = evaluate_model(model, test)
    logger.info(f"Model accuracy: {accuracy}")

    logger.info("Pipeline complete")

if __name__ == "__main__":
    main()
