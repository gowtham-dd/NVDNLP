
from src.NVDNLP.config.configuration import ConfigurationManager
from src.NVDNLP.components.DataValidation import DataValidation
from src.NVDNLP import logger

STAGE_NAME = "Data Validation stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        
        # Check if validation is already completed
        if data_validation.is_validation_completed():
            logger.info("Data validation already completed successfully. Skipping...")
            return True
        
        # Perform validation
        data_path = "artifacts/data_ingestion/nvd_combined_2010_2025.csv"
        validation_status = data_validation.validate_all(data_path)
        
        if validation_status:
            logger.info(" Data Validation completed successfully!")
        else:
            logger.info(" Data Validation failed!")
        
        return validation_status

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    
    except Exception as e:
        logger.exception(e)
        raise e