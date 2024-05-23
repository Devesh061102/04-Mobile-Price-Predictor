from Mobile_Price_Predictor.config.configuration import ConfigurationManager
from Mobile_Price_Predictor.conponents.model_trainer import ModelTrainer
from Mobile_Price_Predictor.logging import logger


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer_config = ModelTrainer(config=model_trainer_config)
        model_trainer_config.train()