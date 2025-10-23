# ============================================
# 🤖 MODEL TRAINER COMPONENT
# ============================================

import os
import joblib
import pandas as pd
import xgboost as xgb
from src.NVDNLP.entity.config_entity import ModelTrainerConfig
from src.NVDNLP import logger

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, train_data, test_data, label_encoder, tfidf_vectorizer):
        self.config = config
        self.train_data = train_data
        self.test_data = test_data
        self.label_encoder = label_encoder
        self.tfidf_vectorizer = tfidf_vectorizer
        self.model = None

    def _check_model_exists(self) -> bool:
        """Check if model already exists in artifacts/model_training"""
        return os.path.exists(self.config.trained_model_path)

    def prepare_data(self):
        """Prepare training and testing data"""
        try:
            logger.info(" Preparing data for model training...")
            
            # Extract features and labels from train data
            X_train_tfidf = self.tfidf_vectorizer.transform(self.train_data['Description'].astype(str))
            y_train = self.train_data['encoded_severity']
            
            # Extract features and labels from test data
            X_test_tfidf = self.tfidf_vectorizer.transform(self.test_data['Description'].astype(str))
            y_test = self.test_data['encoded_severity']
            
            logger.info(f" Training data: {X_train_tfidf.shape}, {len(y_train)} samples")
            logger.info(f" Testing data: {X_test_tfidf.shape}, {len(y_test)} samples")
            
            return X_train_tfidf, X_test_tfidf, y_train, y_test
            
        except Exception as e:
            logger.error(f" Data preparation failed: {e}")
            raise e

    def initialize_model(self):
        """Initialize XGBoost classifier with configuration"""
        try:
            logger.info(" Initializing XGBoost model...")
            
            self.model = xgb.XGBClassifier(
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                max_depth=self.config.max_depth,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                random_state=self.config.random_state,
                tree_method=self.config.tree_method,
                eval_metric=self.config.eval_metric,
                early_stopping_rounds=self.config.early_stopping_rounds
            )
            
            logger.info(" XGBoost model initialized successfully!")
            return self.model
            
        except Exception as e:
            logger.error(f" Model initialization failed: {e}")
            raise e

    def train_model(self, X_train, X_test, y_train, y_test):
        """Train the XGBoost model"""
        try:
            logger.info(" Training tuned XGBoost model...")
            
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=50
            )
            
            logger.info(" Model training completed successfully!")
            return self.model
            
        except Exception as e:
            logger.error(f" Model training failed: {e}")
            raise e

    def save_model(self):
        """Save trained model and artifacts"""
        try:
            logger.info(" Saving model and artifacts...")
            
            # Create directory if it doesn't exist
            os.makedirs(self.config.model_dir, exist_ok=True)
            
            # Save the trained model
            joblib.dump(self.model, self.config.trained_model_path)
            logger.info(f" Saved trained model: {self.config.trained_model_path}")
            
            # Save training configuration
            training_info = {
                'model_type': 'XGBoost',
                'n_estimators': self.config.n_estimators,
                'learning_rate': self.config.learning_rate,
                'max_depth': self.config.max_depth,
                'training_samples': len(self.train_data),
                'feature_dimensions': self.tfidf_vectorizer.transform(['']).shape[1]
            }
            
            info_file = os.path.join(self.config.root_dir, "training_info.txt")
            with open(info_file, 'w') as f:
                f.write("=== MODEL TRAINING INFORMATION ===\n")
                for key, value in training_info.items():
                    f.write(f"{key}: {value}\n")
            
            logger.info(f" Saved training information: {info_file}")
            
        except Exception as e:
            logger.error(f" Failed to save model: {e}")
            raise e

    def train(self):
        """Complete model training pipeline (only training, no evaluation)"""
        try:
            # Check if model already exists
            if self._check_model_exists():
                logger.info(" Model already exists in artifacts/model_training. Skipping training.")
                return {
                    'status': 'skipped',
                    'message': 'Model already exists',
                    'model_path': self.config.trained_model_path
                }
            
            logger.info(" Starting Model Training Pipeline...")
            
            # Step 1: Prepare data
            X_train, X_test, y_train, y_test = self.prepare_data()
            
            # Step 2: Initialize model
            self.initialize_model()
            
            # Step 3: Train model
            self.train_model(X_train, X_test, y_train, y_test)
            
            # Step 4: Save model
            self.save_model()
            
            logger.info(" Model Training completed successfully!")
            
            return {
                'status': 'completed',
                'message': 'Model trained and saved successfully',
                'model_path': self.config.trained_model_path,
                'training_samples': len(self.train_data),
                'test_samples': len(self.test_data)
            }
            
        except Exception as e:
            logger.error(f" Model training pipeline failed: {e}")
            raise e