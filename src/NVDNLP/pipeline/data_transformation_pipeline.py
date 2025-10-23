# ============================================
# 🔄 DATA TRANSFORMATION PIPELINE
# ============================================

from src.NVDNLP.config.configuration import ConfigurationManager
from src.NVDNLP.components.DataTransformation import DataTransformation
from src.NVDNLP import logger

STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        
        # This will automatically skip if already completed
        transformed_data = data_transformation.transform()
        
        return transformed_data

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        result = obj.main()
        
        if result:
            logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        else:
            logger.info(f">>>>>> Stage {STAGE_NAME} skipped (already completed) <<<<<<\n\nx==========x")
    
    except Exception as e:
        logger.exception(e)
        raise e