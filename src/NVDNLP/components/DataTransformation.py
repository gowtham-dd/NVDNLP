# ============================================
# 🔄 DATA TRANSFORMATION COMPONENT
# ============================================

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from src.NVDNLP.entity.config_entity import DataTransformationConfig
from src.NVDNLP import logger

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.label_encoder = None
        self.tfidf_vectorizer = None

    def is_transformation_completed(self) -> bool:
        """Check if data transformation has already been completed by looking for output files"""
        try:
            required_files = [
                self.config.label_encoder_file,
                self.config.tfidf_vectorizer_file,
                self.config.train_file,
                self.config.test_file
            ]
            
            # Check if all required files exist
            all_files_exist = all(os.path.exists(file) for file in required_files)
            
            if all_files_exist:
                logger.info(" Data transformation already completed successfully. Skipping...")
                
                # Verify files can be loaded properly
                try:
                    # Test load one file to ensure it's valid
                    test_encoder = joblib.load(self.config.label_encoder_file)
                    logger.info(" Transformation artifacts verified and valid")
                    return True
                except Exception as e:
                    logger.warning(f" Existing transformation files corrupted. Re-running transformation...")
                    return False
            else:
                logger.info(" Transformation artifacts not found. Starting transformation...")
                return False
                
        except Exception as e:
            logger.error(f" Error checking transformation status: {e}")
            return False

    def load_data(self) -> pd.DataFrame:
        """Load the cleaned dataset"""
        try:
            logger.info(" Loading dataset for transformation...")
            df = pd.read_csv(self.config.data_path)
            logger.info(f" Dataset loaded. Total rows: {len(df)}")
            return df
        except Exception as e:
            logger.info(f" Failed to load dataset: {e}")
            raise e

    def encode_labels(self, severity_series: pd.Series) -> np.ndarray:
        """Encode categorical severity labels to numerical values"""
        try:
            logger.info(" Encoding severity labels...")
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(severity_series)
            logger.info(f" Encoded severity classes: {list(self.label_encoder.classes_)}")
            return y_encoded
        except Exception as e:
            logger.info(f" Label encoding failed: {e}")
            raise e

    def generate_tfidf_features(self, descriptions: pd.Series):
        """Generate TF-IDF features from text descriptions"""
        try:
            logger.info(" Generating enhanced TF-IDF features...")
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.config.max_features,
                ngram_range=self.config.ngram_range,
                stop_words='english'
            )
            X_tfidf = self.tfidf_vectorizer.fit_transform(descriptions.astype(str))
            logger.info(f" TF-IDF shape: {X_tfidf.shape}")
            return X_tfidf
        except Exception as e:
            logger.error(f" TF-IDF feature generation failed: {e}")
            raise e

    def split_data(self, df: pd.DataFrame, y_encoded: np.ndarray):
        """Split data into training and testing sets and return DataFrames"""
        try:
            logger.info(" Splitting data into train and test sets...")
            
            # Split indices
            train_indices, test_indices = train_test_split(
                df.index,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y_encoded
            )
            
            # Create train and test DataFrames
            train_df = df.loc[train_indices].copy()
            test_df = df.loc[test_indices].copy()
            
            # Add encoded labels to DataFrames
            train_df['encoded_severity'] = y_encoded[train_indices]
            test_df['encoded_severity'] = y_encoded[test_indices]
            
            logger.info(f" Train size: {len(train_df)}, Test size: {len(test_df)}")
            return train_df, test_df
            
        except Exception as e:
            logger.error(f" Data splitting failed: {e}")
            raise e

    def save_artifacts(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Save transformed data and transformers"""
        try:
            logger.info(" Saving transformation artifacts...")
            
            # Create directory if it doesn't exist
            os.makedirs(self.config.root_dir, exist_ok=True)
            
            # Save label encoder as PKL
            joblib.dump(self.label_encoder, self.config.label_encoder_file)
            logger.info(f" Saved label encoder: {self.config.label_encoder_file}")
            
            # Save TF-IDF vectorizer as PKL
            joblib.dump(self.tfidf_vectorizer, self.config.tfidf_vectorizer_file)
            logger.info(f" Saved TF-IDF vectorizer: {self.config.tfidf_vectorizer_file}")
            
            # Save train and test as CSV files
            train_df.to_csv(self.config.train_file, index=False)
            test_df.to_csv(self.config.test_file, index=False)
            
            logger.info(f" Saved train data (CSV): {self.config.train_file}")
            logger.info(f" Saved test data (CSV): {self.config.test_file}")
            
            # Create a status file to mark completion
            status_file = os.path.join(self.config.root_dir, "transformation_status.txt")
            with open(status_file, 'w') as f:
                f.write("Data Transformation Status: COMPLETED\n")
                f.write(f"Label Encoder: {self.config.label_encoder_file}\n")
                f.write(f"TF-IDF Vectorizer: {self.config.tfidf_vectorizer_file}\n")
                f.write(f"Train Data: {self.config.train_file}\n")
                f.write(f"Test Data: {self.config.test_file}\n")
                f.write(f"Train Samples: {len(train_df)}\n")
                f.write(f"Test Samples: {len(test_df)}\n")
            
        except Exception as e:
            logger.error(f" Failed to save artifacts: {e}")
            raise e

    def load_existing_artifacts(self):
        """Load existing transformation artifacts"""
        try:
            logger.info(" Loading existing transformation artifacts...")
            
            # Load transformers (PKL files)
            self.label_encoder = joblib.load(self.config.label_encoder_file)
            self.tfidf_vectorizer = joblib.load(self.config.tfidf_vectorizer_file)
            
            # Load train and test data (CSV files)
            train_df = pd.read_csv(self.config.train_file)
            test_df = pd.read_csv(self.config.test_file)
            
            logger.info(" Successfully loaded existing transformation artifacts")
            
            return {
                'train_df': train_df,
                'test_df': test_df,
                'label_encoder': self.label_encoder,
                'tfidf_vectorizer': self.tfidf_vectorizer
            }
            
        except Exception as e:
            logger.error(f" Failed to load existing artifacts: {e}")
            raise e

    def transform(self):
        """Perform complete data transformation pipeline only if not already completed"""
        try:
            # Check if transformation is already completed
            if self.is_transformation_completed():
                return self.load_existing_artifacts()
            
            logger.info(" Starting Data Transformation Pipeline...")
            
            # Step 1: Load data
            df = self.load_data()
            
            # Step 2: Prepare features and target
            X = df["Description"].astype(str)
            y = df["Severity"]
            
            # Step 3: Encode labels
            y_encoded = self.encode_labels(y)
            
            # Step 4: Generate TF-IDF features
            X_tfidf = self.generate_tfidf_features(X)
            
            # Step 5: Split data into train and test DataFrames
            train_df, test_df = self.split_data(df, y_encoded)
            
            # Step 6: Save artifacts
            self.save_artifacts(train_df, test_df)
            
            logger.info(" Data Transformation completed successfully!")
            
            return {
                'train_df': train_df,
                'test_df': test_df,
                'label_encoder': self.label_encoder,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'X_tfidf': X_tfidf,
                'y_encoded': y_encoded
            }
            
        except Exception as e:
            logger.error(f" Data transformation failed: {e}")
            raise e