# ============================================
#     MODEL EVALUATION PIPELINE
# ============================================

from src.NVDNLP.config.configuration import ConfigurationManager
from src.NVDNLP.components.ModelEvaluation import ModelEvaluation
from src.NVDNLP import logger

STAGE_NAME = "Model Evaluation stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        
        # Evaluate model (will skip if metrics already exist)
        evaluation_result = model_evaluation.evaluate()
        
        return evaluation_result

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationTrainingPipeline()
        result = obj.main()
        
        if result['status'] == 'completed':
            logger.info(f" Evaluation completed successfully!")
            logger.info(f" Accuracy: {result['accuracy']*100:.2f}%")
            logger.info(f" Test samples: {result['test_samples']}")
            logger.info(f" Metrics saved at: {result['metrics_file']}")
            logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        else:
            logger.info(f">>>>>> Stage {STAGE_NAME} skipped (evaluation already completed) <<<<<<\n\nx==========x")
    
    except Exception as e:
        logger.exception(e)
        raise e