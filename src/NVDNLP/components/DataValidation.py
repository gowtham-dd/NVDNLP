
import os
import pandas as pd
from NVDNLP import logger
from src.NVDNLP.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
    
    def is_validation_completed(self) -> bool:
        """Check if validation has already been completed by looking for status file"""
        try:
            if os.path.exists(self.config.STATUS_FILE):
                with open(self.config.STATUS_FILE, 'r') as f:
                    status_content = f.read().strip()
                    if "Validation status: True" in status_content:
                        logger.info(" Data validation already completed successfully. Skipping...")
                        return True
                    elif "Validation status: False" in status_content:
                        logger.info("Previous validation failed. Re-running validation...")
                        return False
            return False
        except Exception as e:
            logger.info(f"Error checking validation status: {e}")
            return False
    
    def validate_all_files_exist(self) -> bool:
        """Validate that all required files exist"""
        try:
            validation_status = True
            
            # Check if all required files exist
            for file in self.config.ALL_REQUIRED_FILES:
                file_path = os.path.join("artifacts", "data_ingestion", file)
                if not os.path.exists(file_path):
                    validation_status = False
                    logger.info(f"Missing required file: {file}")
                    break
            
            # Write validation status to file
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")
            
            return validation_status
        
        except Exception as e:
            logger.info(f"Error in file validation: {e}")
            raise e
    
    def validate_dataset_columns(self, data_path: str) -> bool:
        """Validate that dataset contains all required columns"""
        try:
            validation_status = True
            
            # Read the dataset
            df = pd.read_csv(data_path)
            
            # Check if all required columns are present
            missing_columns = []
            for column in self.config.REQUIRED_COLUMNS:
                if column not in df.columns:
                    missing_columns.append(column)
                    validation_status = False
            
            if missing_columns:
                logger.info(f"Missing required columns: {missing_columns}")
            else:
                logger.info(" All required columns are present!")
                
                # Additional validation: Check for null values in critical columns
                critical_columns = ['Description', 'Severity']
                null_check_status = True
                for col in critical_columns:
                    null_count = df[col].isnull().sum()
                    if null_count > 0:
                        logger.info(f"Column '{col}' has {null_count} null values")
                        null_check_status = False
                
                if null_check_status:
                    logger.info(" No null values in critical columns!")
                else:
                    logger.info("  Null values found in critical columns")
            
            # Write detailed validation report
            validation_report = {
                'validation_status': validation_status,
                'missing_columns': missing_columns,
                'total_rows': len(df),
                'columns_present': list(df.columns),
                'severity_distribution': df['Severity'].value_counts().to_dict() if 'Severity' in df.columns else {}
            }
            
            # Save validation report
            report_path = os.path.join(self.config.root_dir, "validation_report.txt")
            with open(report_path, 'w') as f:
                f.write("=== DATA VALIDATION REPORT ===\n")
                f.write(f"Overall Status: {'PASS' if validation_status else 'FAIL'}\n")
                f.write(f"Total Rows: {validation_report['total_rows']}\n")
                f.write(f"Missing Columns: {missing_columns}\n")
                f.write(f"Columns Present: {validation_report['columns_present']}\n")
                f.write(f"Severity Distribution: {validation_report['severity_distribution']}\n")
            
            # Update status file
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")
            
            return validation_status
        
        except Exception as e:
            logger.info(f"Error in dataset validation: {e}")
            raise e
    
    def validate_all(self, data_path: str) -> bool:
        """Perform complete data validation only if not already completed"""
        try:
            # Check if validation is already completed
            if self.is_validation_completed():
                return True
            
            logger.info(" Starting Data Validation Process...")
            
            # Validate files exist
            files_valid = self.validate_all_files_exist()
            
            # Validate dataset structure
            dataset_valid = self.validate_dataset_columns(data_path)
            
            # Overall validation status
            overall_status = files_valid and dataset_valid
            
            if overall_status:
                logger.info(" Data Validation PASSED - All checks completed successfully!")
            else:
                logger.info(" Data Validation FAILED - Check validation report for details!")
            
            return overall_status
            
        except Exception as e:
            logger.info(f"Error in complete validation: {e}")
            # Mark as failed in status file
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write("Validation status: False")
            raise e