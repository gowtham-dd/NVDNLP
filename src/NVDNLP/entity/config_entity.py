## ENTITY
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    source_URL:str
    local_data_file:Path
    unzip_dir:Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: list
    REQUIRED_COLUMNS: list



@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    train_file: Path
    test_file: Path
    label_encoder_file: Path
    tfidf_vectorizer_file: Path
    test_size: float
    random_state: int
    max_features: int
    ngram_range: tuple



@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    model_dir: Path
    trained_model_path: Path
    n_estimators: int
    learning_rate: float
    max_depth: int
    subsample: float
    colsample_bytree: float
    random_state: int
    tree_method: str
    eval_metric: str
    early_stopping_rounds: int



@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    label_encoder_path: Path
    tfidf_vectorizer_path: Path
    metric_file_name: Path



@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    label_encoder_path: Path
    tfidf_vectorizer_path: Path
    metric_file_name: Path