
from src.NVDNLP import logger

from src.NVDNLP.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.NVDNLP.pipeline.data_validation_pipeline import DataValidationTrainingPipeline
from src.NVDNLP.pipeline.data_transformation_pipeline import DataTransformationTrainingPipeline
from src.NVDNLP.pipeline.model_training_pipeline import ModelTrainerTrainingPipeline




STAGE_NAME="Data Ingestion stage"


try:
    logger.info(f">>>> Stage {STAGE_NAME} started")
    obj=DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>> Stage {STAGE_NAME} completed")

except Exception as e:
    logger.exception(e)
    raise e





STAGE_NAME="Data Validation stage"


try:
    logger.info(f">>>> Stage {STAGE_NAME} started")
    obj=DataValidationTrainingPipeline()
    obj.main()
    logger.info(f">>>>> Stage {STAGE_NAME} completed")

except Exception as e:
    logger.exception(e)
    raise e





STAGE_NAME="Data Transformation stage"


try:
    logger.info(f">>>> Stage {STAGE_NAME} started")
    obj=DataTransformationTrainingPipeline()
    obj.main()
    logger.info(f">>>>> Stage {STAGE_NAME} completed")

except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME="Model Training stage"


try:
    logger.info(f">>>> Stage {STAGE_NAME} started")
    obj=ModelTrainerTrainingPipeline()
    obj.main()
    logger.info(f">>>>> Stage {STAGE_NAME} completed")

except Exception as e:
    logger.exception(e)
    raise e





STAGE_NAME="Model Evaluation stage"


try:
    logger.info(f">>>> Stage {STAGE_NAME} started")
    obj=ModelTrainerTrainingPipeline()
    obj.main()
    logger.info(f">>>>> Stage {STAGE_NAME} completed")

except Exception as e:
    logger.exception(e)
    raise e

