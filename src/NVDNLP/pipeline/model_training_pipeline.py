# ============================================
# 🤖 MODEL TRAINING PIPELINE
# ============================================

from src.NVDNLP.config.configuration import ConfigurationManager
from src.NVDNLP.components.DataTransformation import DataTransformation
from src.NVDNLP.components.ModelTraining import ModelTrainer
from src.NVDNLP import logger

STAGE_NAME = "Model Trainer stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        
        # Load data transformation artifacts
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        transformation_result = data_transformation.transform()
        
        # Get model trainer config
        model_trainer_config = config.get_model_trainer_config()
        
        # Initialize and train model
        model_trainer = ModelTrainer(
            config=model_trainer_config,
            train_data=transformation_result['train_df'],
            test_data=transformation_result['test_df'],
            label_encoder=transformation_result['label_encoder'],
            tfidf_vectorizer=transformation_result['tfidf_vectorizer']
        )
        
        # Train model (will skip if already exists)
        training_result = model_trainer.train()
        
        return training_result

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainerTrainingPipeline()
        result = obj.main()
        
        if result['status'] == 'completed':
            logger.info(f" Model trained successfully!")
            logger.info(f" Training samples: {result['training_samples']}")
            logger.info(f" Test samples: {result['test_samples']}")
            logger.info(f" Model saved at: {result['model_path']}")
            logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        else:
            logger.info(f">>>>>> Stage {STAGE_NAME} skipped (model already exists) <<<<<<\n\nx==========x")
    
    except Exception as e:
        logger.exception(e)
        raise e