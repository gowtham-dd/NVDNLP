import os
from NVDNLP.utils.common import read_yaml, create_directories
from src.NVDNLP.constant import *
from src.NVDNLP.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig,ModelEvaluationConfig

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
     ):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES,
            REQUIRED_COLUMNS=config.REQUIRED_COLUMNS,
        )

        return data_validation_config

    

    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        params = self.params.TFIDF

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            train_file=Path(config.train_file),
            test_file=Path(config.test_file),
            label_encoder_file=Path(config.label_encoder_file),
            tfidf_vectorizer_file=Path(config.tfidf_vectorizer_file),
            test_size=self.params.training.test_size,
            random_state=self.params.training.random_state,
            max_features=params.max_features,
            ngram_range=tuple(params.ngram_range),
        )

        return data_transformation_config
    


    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.XGBoost

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            model_dir=Path(config.model_dir),
            trained_model_path=Path(config.trained_model_path),
            n_estimators=params.n_estimators,
            learning_rate=params.learning_rate,
            max_depth=params.max_depth,
            subsample=params.subsample,
            colsample_bytree=params.colsample_bytree,
            random_state=params.random_state,
            tree_method=params.tree_method,
            eval_metric=params.eval_metric,
            early_stopping_rounds=params.early_stopping_rounds
        )

        return model_trainer_config
    


    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            test_data_path=Path(config.test_data_path),
            model_path=Path(config.model_path),
            label_encoder_path=Path(config.label_encoder_path),
            tfidf_vectorizer_path=Path(config.tfidf_vectorizer_path),
            metric_file_name=Path(config.metric_file_name)
        )

        return model_evaluation_config
    

