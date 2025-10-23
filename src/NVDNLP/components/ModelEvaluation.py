# ============================================
#     MODEL EVALUATION COMPONENT
# ============================================

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.NVDNLP.entity.config_entity import ModelEvaluationConfig
from src.NVDNLP import logger

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.model = None
        self.label_encoder = None
        self.tfidf_vectorizer = None

    def load_artifacts(self):
        """Load model, label encoder, and TF-IDF vectorizer"""
        try:
            logger.info(" Loading evaluation artifacts...")
            
            # Load trained model
            self.model = joblib.load(self.config.model_path)
            logger.info(f" Model loaded: {self.config.model_path}")
            
            # Load label encoder
            self.label_encoder = joblib.load(self.config.label_encoder_path)
            logger.info(f" Label encoder loaded: {self.config.label_encoder_path}")
            
            # Load TF-IDF vectorizer
            self.tfidf_vectorizer = joblib.load(self.config.tfidf_vectorizer_path)
            logger.info(f" TF-IDF vectorizer loaded: {self.config.tfidf_vectorizer_path}")
            
        except Exception as e:
            logger.error(f" Failed to load artifacts: {e}")
            raise e

    def load_test_data(self):
        """Load and prepare test data"""
        try:
            logger.info(" Loading test data...")
            
            # Load test CSV file
            test_df = pd.read_csv(self.config.test_data_path)
            logger.info(f" Test data loaded: {len(test_df)} samples")
            
            # Prepare features and labels
            X_test_descriptions = test_df['Description'].astype(str)
            y_test_encoded = test_df['encoded_severity']
            
            # Transform descriptions to TF-IDF features
            X_test_tfidf = self.tfidf_vectorizer.transform(X_test_descriptions)
            logger.info(f" Test features transformed: {X_test_tfidf.shape}")
            
            return X_test_tfidf, y_test_encoded, test_df
            
        except Exception as e:
            logger.error(f" Failed to load test data: {e}")
            raise e

    def make_predictions(self, X_test):
        """Make predictions on test data"""
        try:
            logger.info(" Making predictions on test data...")
            
            y_pred = self.model.predict(X_test)
            logger.info(f" Predictions completed: {len(y_pred)} predictions")
            
            return y_pred
            
        except Exception as e:
            logger.error(f" Prediction failed: {e}")
            raise e

    def calculate_metrics(self, y_test, y_pred):
        """Calculate evaluation metrics"""
        try:
            logger.info(" Calculating evaluation metrics...")
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            # Generate classification report
            class_report = classification_report(
                y_test, 
                y_pred, 
                target_names=self.label_encoder.classes_,
                output_dict=True
            )
            
            # Generate confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Convert confusion matrix to list for JSON serialization
            conf_matrix_list = conf_matrix.tolist()
            
            logger.info(f" Accuracy: {accuracy*100:.2f}%")
            
            return {
                'accuracy': accuracy,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix_list,
                'severity_classes': list(self.label_encoder.classes_)
            }
            
        except Exception as e:
            logger.error(f" Metric calculation failed: {e}")
            raise e

    def save_metrics(self, metrics):
        """Save evaluation metrics to file"""
        try:
            logger.info(" Saving evaluation metrics...")
            
            # Create detailed metrics report
            metrics_report = {
                'overall_accuracy': metrics['accuracy'],
                'severity_classes': metrics['severity_classes'],
                'confusion_matrix': metrics['confusion_matrix'],
                'detailed_classification_report': metrics['classification_report']
            }
            
            # Save as CSV for easy viewing
            csv_metrics = {
                'metric': ['accuracy'],
                'value': [metrics['accuracy']]
            }
            
            # Add per-class metrics
            for class_name in self.label_encoder.classes_:
                if class_name in metrics['classification_report']:
                    class_metrics = metrics['classification_report'][class_name]
                    csv_metrics['metric'].extend([
                        f'{class_name}_precision',
                        f'{class_name}_recall', 
                        f'{class_name}_f1_score',
                        f'{class_name}_support'
                    ])
                    csv_metrics['value'].extend([
                        class_metrics['precision'],
                        class_metrics['recall'],
                        class_metrics['f1-score'],
                        class_metrics['support']
                    ])
            
            # Create DataFrame and save as CSV
            metrics_df = pd.DataFrame(csv_metrics)
            metrics_df.to_csv(self.config.metric_file_name, index=False)
            logger.info(f" Metrics saved to: {self.config.metric_file_name}")
            
            # Save detailed report as JSON
            import json
            detailed_metrics_file = self.config.metric_file_name.with_suffix('.json')
            with open(detailed_metrics_file, 'w') as f:
                json.dump(metrics_report, f, indent=4)
            logger.info(f" Detailed metrics saved to: {detailed_metrics_file}")
            
        except Exception as e:
            logger.error(f" Failed to save metrics: {e}")
            raise e

    def print_evaluation_summary(self, metrics, y_test, y_pred):
        """Print comprehensive evaluation summary"""
        try:
            logger.info("\n" + "="*60)
            logger.info(" MODEL EVALUATION SUMMARY")
            logger.info("="*60)
            logger.info(f" Overall Accuracy: {metrics['accuracy']*100:.2f}%")
            logger.info(f" Test Samples: {len(y_test)}")
            logger.info(f" Severity Classes: {', '.join(metrics['severity_classes'])}")
            
            logger.info("\n Classification Report:")
            logger.info(classification_report(
                y_test, 
                y_pred, 
                target_names=metrics['severity_classes']
            ))
            
            logger.info("\n Confusion Matrix:")
            logger.info(np.array2string(
                np.array(metrics['confusion_matrix']), 
                formatter={'int': lambda x: f'{x:4d}'}
            ))
            
        except Exception as e:
            logger.error(f" Failed to print evaluation summary: {e}")
            raise e

    def evaluate(self):
        """Complete model evaluation pipeline"""
        try:
            # Check if metrics file already exists
            if self.config.metric_file_name.exists():
                logger.info(" Metrics file already exists. Skipping evaluation...")
                return {
                    'status': 'skipped',
                    'message': 'Evaluation already completed',
                    'metrics_file': self.config.metric_file_name
                }
            
            logger.info(" Starting Model Evaluation Pipeline...")
            
            # Step 1: Load artifacts
            self.load_artifacts()
            
            # Step 2: Load and prepare test data
            X_test, y_test, test_df = self.load_test_data()
            
            # Step 3: Make predictions
            y_pred = self.make_predictions(X_test)
            
            # Step 4: Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred)
            
            # Step 5: Save metrics
            self.save_metrics(metrics)
            
            # Step 6: Print summary
            self.print_evaluation_summary(metrics, y_test, y_pred)
            
            logger.info(" Model Evaluation completed successfully!")
            
            return {
                'status': 'completed',
                'message': 'Evaluation completed successfully',
                'accuracy': metrics['accuracy'],
                'metrics_file': self.config.metric_file_name,
                'test_samples': len(y_test)
            }
            
        except Exception as e:
            logger.error(f" Model evaluation pipeline failed: {e}")
            raise e