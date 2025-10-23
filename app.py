# ============================================
# 🚀 FLASK WEB APPLICATION (ENHANCED JSON PROCESSING)
# ============================================

from flask import Flask, render_template, request, jsonify
import os
import joblib
import pandas as pd
import json
import numpy as np
from datetime import datetime
import logging
import re
import time
from threading import Thread, Event
from queue import Queue
import uuid

app = Flask(__name__)
app.secret_key = 'nvd-severity-classifier-secret-key'

# Add these global variables after app initialization
processing_tasks = {}
stop_events = {}
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessingThread(Thread):
    def __init__(self, task_id, descriptions_data, predictor, callback):
        Thread.__init__(self)
        self.task_id = task_id
        self.descriptions_data = descriptions_data
        self.predictor = predictor
        self.callback = callback
        self.stop_event = Event()
        self.results = []
        self.processed_count = 0
        self.total_count = len(descriptions_data)
        
    def run(self):
        try:
            batch_size = 100  # Process in batches of 100
            for i in range(0, len(self.descriptions_data), batch_size):
                if self.stop_event.is_set():
                    self.callback({
                        'task_id': self.task_id,
                        'status': 'stopped',
                        'processed': self.processed_count,
                        'total': self.total_count,
                        'results': self.results
                    })
                    return
                
                batch_descriptions = self.descriptions_data[i:i + batch_size]
                texts = [item['description'] for item in batch_descriptions]
                
                # Process batch
                batch_predictions = self.predictor.predict_batch(texts)
                
                # Combine with extracted data
                for j, pred in enumerate(batch_predictions):
                    result = {
                        **pred,
                        'cve_id': batch_descriptions[j]['cve_id'],
                        'product': batch_descriptions[j]['product'],
                        'cvss_score': batch_descriptions[j]['cvss_score'],
                        'cvss_severity': batch_descriptions[j]['cvss_severity'],
                        'published_date': batch_descriptions[j]['published_date'],
                        'priority': len(self.results) + 1
                    }
                    self.results.append(result)
                
                self.processed_count = i + len(batch_descriptions)
                
                # Send progress update
                self.callback({
                    'task_id': self.task_id,
                    'status': 'processing',
                    'processed': self.processed_count,
                    'total': self.total_count,
                    'results': self.results[-len(batch_descriptions):]  # Only new results
                })
                
                # Small delay to prevent overwhelming the browser
                time.sleep(0.1)
            
            # Processing completed
            self.callback({
                'task_id': self.task_id,
                'status': 'completed',
                'processed': self.processed_count,
                'total': self.total_count,
                'results': self.results
            })
            
        except Exception as e:
            self.callback({
                'task_id': self.task_id,
                'status': 'error',
                'error': str(e),
                'processed': self.processed_count,
                'total': self.total_count
            })

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# Use the custom encoder
app.json_encoder = NumpyEncoder

class SeverityPredictor:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.tfidf_vectorizer = None
        self.load_artifacts()
    
    def load_artifacts(self):
        """Load the trained model and transformers"""
        try:
            # Try multiple possible paths for model
            model_paths = [
                'artifacts/model_training/xgb_severity_model.pkl',  # Your actual path
                'artifacts/model_training/model/xgboost_severity_model.pkl',
                'artifacts/model_training/model.pkl'
            ]
            
            for model_path in model_paths:
                if os.path.exists(model_path):
                    self.model = joblib.load(model_path)
                    logger.info(f"✅ Model loaded from: {model_path}")
                    break
            
            if self.model is None:
                logger.error("❌ Model file not found in any expected location")
                return False
            
            # Load label encoder
            le_paths = [
                'artifacts/data_transformation/label_encoder.pkl',
                'artifacts/data_transformation/label_encoder.pkl'
            ]
            
            for le_path in le_paths:
                if os.path.exists(le_path):
                    self.label_encoder = joblib.load(le_path)
                    logger.info(f"✅ Label encoder loaded from: {le_path}")
                    logger.info(f"🏷️ Label encoder classes: {list(self.label_encoder.classes_)}")
                    break
            
            # Load TF-IDF vectorizer
            tfidf_paths = [
                'artifacts/data_transformation/tfidf_vectorizer.pkl',
                'artifacts/data_transformation/tfidf_vectorizer.pkl'
            ]
            
            for tfidf_path in tfidf_paths:
                if os.path.exists(tfidf_path):
                    self.tfidf_vectorizer = joblib.load(tfidf_path)
                    logger.info(f"✅ TF-IDF vectorizer loaded from: {tfidf_path}")
                    break
            
            if all([self.model, self.label_encoder, self.tfidf_vectorizer]):
                logger.info("✅ All artifacts loaded successfully!")
                return True
            else:
                missing = []
                if not self.model: missing.append("model")
                if not self.label_encoder: missing.append("label_encoder")
                if not self.tfidf_vectorizer: missing.append("tfidf_vectorizer")
                logger.error(f"❌ Missing artifacts: {', '.join(missing)}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to load artifacts: {e}")
            return False
    
    def convert_to_serializable(self, obj):
        """Convert NumPy types to Python native types for JSON serialization"""
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj
    
    def predict_single_text(self, text):
        """Predict severity for a single text"""
        try:
            # Transform text to TF-IDF features
            text_tfidf = self.tfidf_vectorizer.transform([text])
            
            # Make prediction
            prediction = self.model.predict(text_tfidf)
            
            # Convert back to severity label
            severity = self.label_encoder.inverse_transform(prediction)[0]
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(text_tfidf)[0]
            confidence = max(probabilities)
            
            # Convert to serializable types
            confidence = self.convert_to_serializable(confidence)
            
            return {
                'text': text,
                'severity': severity,
                'confidence': round(confidence * 100, 2),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'success': True
            }
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return {
                'text': text,
                'error': str(e),
                'success': False
            }
    
    def predict_batch(self, texts):
        """Predict severity for multiple texts"""
        try:
            # Transform texts to TF-IDF features
            texts_tfidf = self.tfidf_vectorizer.transform(texts)
            
            # Make predictions
            predictions = self.model.predict(texts_tfidf)
            
            # Convert back to severity labels
            severities = self.label_encoder.inverse_transform(predictions)
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(texts_tfidf)
            confidences = [max(probs) * 100 for probs in probabilities]
            
            results = []
            for i, text in enumerate(texts):
                # Convert to serializable types
                confidence = self.convert_to_serializable(confidences[i])
                
                results.append({
                    'text': text,
                    'severity': severities[i],
                    'confidence': round(confidence, 2),
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'success': True
                })
            
            return results
        except Exception as e:
            logger.error(f"❌ Batch prediction failed: {e}")
            raise e
    
    def extract_product_from_description(self, description):
        """Extract product/vendor name from vulnerability description"""
        try:
            # Common patterns for product extraction
            patterns = [
                # Pattern: "Product X through Y" or "Product X version Y"
                r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(?:through|version|v|\d)',
                # Pattern: "in Product X" 
                r'in\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-z]+)*)',
                # Pattern: "Product X allows"
                r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+allows',
                # Pattern: "Affects Product X"
                r'[Aa]ffects\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, description)
                if match:
                    product = match.group(1).strip()
                    # Filter out common false positives
                    common_false_positives = ['The', 'This', 'A', 'An', 'Allows', 'Through', 'Version']
                    if product not in common_false_positives and len(product) > 2:
                        return product
            
            # If no pattern matches, try to extract first few capitalized words
            words = description.split()
            product_words = []
            for word in words[:6]:  # Check first 6 words
                if word.istitle() and len(word) > 2 and word not in ['The', 'This', 'A', 'An', 'In', 'On', 'At']:
                    product_words.append(word)
                elif product_words:  # Stop if we hit a non-capitalized word after finding some
                    break
            
            if product_words:
                return ' '.join(product_words[:3])  # Return max 3 words
            
            return "Unknown Product"
            
        except Exception as e:
            logger.error(f"❌ Failed to extract product from description: {e}")
            return "Unknown Product"
    
    def extract_descriptions_from_json(self, json_data):
        """Extract vulnerability descriptions from NVD JSON file with product information"""
        try:
            descriptions = []
            
            # Handle different JSON structures
            if isinstance(json_data, list):
                # JSON is a list of vulnerabilities
                for item in json_data:
                    self._process_vulnerability_item(item, descriptions)
                    
            elif 'vulnerabilities' in json_data:
                # Standard NVD API response format
                for item in json_data['vulnerabilities']:
                    self._process_vulnerability_item(item, descriptions)
                    
            elif 'CVE_Items' in json_data:
                # Legacy NVD format
                for item in json_data['CVE_Items']:
                    self._process_legacy_vulnerability_item(item, descriptions)
                    
            else:
                # Try to find vulnerabilities in root level
                for key, value in json_data.items():
                    if isinstance(value, list) and len(value) > 0:
                        if any('cve' in str(item) or 'CVE' in str(item) for item in value[:2]):
                            for item in value:
                                self._process_vulnerability_item(item, descriptions)
            
            logger.info(f"📊 Extracted {len(descriptions)} vulnerability descriptions")
            return descriptions
            
        except Exception as e:
            logger.error(f"❌ Failed to extract descriptions from JSON: {e}")
            raise e
    
    def _process_vulnerability_item(self, item, descriptions):
        """Process a single vulnerability item from modern NVD JSON format"""
        try:
            cve_data = item.get('cve', {})
            
            # Extract CVE ID
            cve_id = cve_data.get('id', 'Unknown')
            
            # Extract description (prefer English)
            descriptions_list = cve_data.get('descriptions', [])
            english_description = None
            fallback_description = None
            
            for desc in descriptions_list:
                if desc.get('lang') == 'en':
                    english_description = desc.get('value', '')
                    break
                elif not fallback_description:
                    fallback_description = desc.get('value', '')
            
            description = english_description or fallback_description or 'No description available'
            
            # Extract CVSS score if available
            metrics = cve_data.get('metrics', {})
            cvss_score = None
            cvss_severity = None
            
            # Try CVSS v3.1 first, then v3.0, then v2.0
            for cvss_version in ['cvssMetricV31', 'cvssMetricV30', 'cvssMetricV2']:
                if cvss_version in metrics:
                    cvss_data = metrics[cvss_version][0].get('cvssData', {})
                    cvss_score = cvss_data.get('baseScore')
                    cvss_severity = cvss_data.get('baseSeverity')
                    if cvss_score:
                        break
            
            # Extract product/vendor from description
            product = self.extract_product_from_description(description)
            
            # Extract published date
            published = cve_data.get('published', '')
            
            descriptions.append({
                'cve_id': cve_id,
                'description': description,
                'product': product,
                'cvss_score': cvss_score,
                'cvss_severity': cvss_severity,
                'published_date': published
            })
            
        except Exception as e:
            logger.error(f"❌ Failed to process vulnerability item: {e}")
    
    def _process_legacy_vulnerability_item(self, item, descriptions):
        """Process a single vulnerability item from legacy NVD JSON format"""
        try:
            cve_data = item.get('cve', {})
            cve_meta = cve_data.get('CVE_data_meta', {})
            
            # Extract CVE ID
            cve_id = cve_meta.get('ID', 'Unknown')
            
            # Extract description
            description_data = cve_data.get('description', {}).get('description_data', [])
            description = description_data[0].get('value', 'No description available') if description_data else 'No description available'
            
            # Extract product/vendor from description
            product = self.extract_product_from_description(description)
            
            descriptions.append({
                'cve_id': cve_id,
                'description': description,
                'product': product,
                'cvss_score': None,
                'cvss_severity': None,
                'published_date': ''
            })
            
        except Exception as e:
            logger.error(f"❌ Failed to process legacy vulnerability item: {e}")

# Initialize predictor
try:
    predictor = SeverityPredictor()
    if predictor.model and predictor.label_encoder and predictor.tfidf_vectorizer:
        logger.info("✅ Severity predictor initialized successfully!")
    else:
        logger.error("❌ Severity predictor initialization failed - missing artifacts")
        predictor = None
except Exception as e:
    logger.error(f"❌ Failed to initialize predictor: {e}")
    predictor = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_single', methods=['POST'])
def predict_single():
    """Handle single text prediction"""
    try:
        if predictor is None:
            return jsonify({'error': 'Predictor not initialized. Please check if model files exist.', 'success': False}), 500
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received', 'success': False}), 400
            
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided', 'success': False}), 400
        
        logger.info(f"🔍 Analyzing text: {text[:100]}...")
        
        # Make prediction
        result = predictor.predict_single_text(text)
        
        logger.info(f"✅ Prediction result: {result['severity']} with {result['confidence']}% confidence")
        
        # Ensure all values are JSON serializable
        serializable_result = {}
        for key, value in result.items():
            serializable_result[key] = predictor.convert_to_serializable(value)
        
        return jsonify(serializable_result)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        error_result = {
            'error': str(e),
            'success': False,
            'text': data.get('text', '') if 'data' in locals() else ''
        }
        return jsonify(error_result), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Handle batch prediction from JSON file"""
    try:
        if predictor is None:
            return jsonify({'error': 'Predictor not initialized', 'success': False}), 500
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded', 'success': False}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected', 'success': False}), 400
        
        if not file.filename.endswith('.json'):
            return jsonify({'error': 'Please upload a JSON file', 'success': False}), 400
        
        logger.info(f"📁 Processing file: {file.filename}")
        
        # Read and parse JSON file
        json_data = json.load(file)
        
        # Extract descriptions from JSON
        descriptions_data = predictor.extract_descriptions_from_json(json_data)
        
        if not descriptions_data:
            return jsonify({'error': 'No vulnerability descriptions found in the file. Please check the JSON format.', 'success': False}), 400
        
        logger.info(f"🔍 Found {len(descriptions_data)} vulnerabilities to analyze")
        
        # Extract just the description texts for prediction
        texts = [item['description'] for item in descriptions_data]
        
        # Make batch predictions
        predictions = predictor.predict_batch(texts)
        
        # Combine with extracted data
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                **pred,
                'cve_id': descriptions_data[i]['cve_id'],
                'product': descriptions_data[i]['product'],
                'cvss_score': descriptions_data[i]['cvss_score'],
                'cvss_severity': descriptions_data[i]['cvss_severity'],
                'published_date': descriptions_data[i]['published_date']
            })
        
        # Sort by severity priority (High > Medium > Low)
        severity_order = {'High': 0, 'Medium': 1, 'Low': 2}
        results.sort(key=lambda x: severity_order.get(x['severity'], 3))
        
        # Add priority numbers
        for i, result in enumerate(results):
            result['priority'] = i + 1
        
        logger.info(f"✅ Batch processing completed: {len(results)} results")
        
        # Ensure all values are JSON serializable
        serializable_results = []
        for result in results:
            serializable_result = {}
            for key, value in result.items():
                serializable_result[key] = predictor.convert_to_serializable(value)
            serializable_results.append(serializable_result)
        
        return jsonify({
            'success': True,
            'total_vulnerabilities': len(serializable_results),
            'severity_counts': {
                'High': len([r for r in serializable_results if r['severity'] == 'High']),
                'Medium': len([r for r in serializable_results if r['severity'] == 'Medium']),
                'Low': len([r for r in serializable_results if r['severity'] == 'Low'])
            },
            'results': serializable_results
        })
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/start_batch_processing', methods=['POST'])
def start_batch_processing():
    """Start batch processing in a separate thread"""
    try:
        if predictor is None:
            return jsonify({'error': 'Predictor not initialized', 'success': False}), 500
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded', 'success': False}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected', 'success': False}), 400
        
        if not file.filename.endswith('.json'):
            return jsonify({'error': 'Please upload a JSON file', 'success': False}), 400
        
        # Read and parse JSON file
        json_data = json.load(file)
        
        # Extract descriptions from JSON
        descriptions_data = predictor.extract_descriptions_from_json(json_data)
        
        if not descriptions_data:
            return jsonify({'error': 'No vulnerability descriptions found in the file', 'success': False}), 400
        
        # Create task
        task_id = str(uuid.uuid4())
        
        def progress_callback(update):
            processing_tasks[task_id] = update
        
        # Create and start processing thread
        thread = ProcessingThread(task_id, descriptions_data, predictor, progress_callback)
        stop_events[task_id] = thread.stop_event
        thread.start()
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'total_vulnerabilities': len(descriptions_data),
            'message': f'Started processing {len(descriptions_data)} vulnerabilities'
        })
        
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/get_processing_status/<task_id>', methods=['GET'])
def get_processing_status(task_id):
    """Get processing status for a task"""
    if task_id in processing_tasks:
        return jsonify(processing_tasks[task_id])
    else:
        return jsonify({'error': 'Task not found', 'success': False}), 404

@app.route('/stop_processing/<task_id>', methods=['POST'])
def stop_processing(task_id):
    """Stop a processing task"""
    if task_id in stop_events:
        stop_events[task_id].set()
        return jsonify({'success': True, 'message': 'Processing stopped'})
    else:
        return jsonify({'error': 'Task not found', 'success': False}), 404

@app.route('/clear_results', methods=['POST'])
def clear_results():
    """Clear all results"""
    processing_tasks.clear()
    stop_events.clear()
    return jsonify({'success': True, 'message': 'Results cleared'})



@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    status = 'healthy' if predictor is not None else 'unhealthy'
    artifacts_loaded = all([predictor.model, predictor.label_encoder, predictor.tfidf_vectorizer]) if predictor else False
    
    return jsonify({
        'status': status,
        'artifacts_loaded': artifacts_loaded,
        'predictor_initialized': predictor is not None
    })

if __name__ == '__main__':
    print("🚀 Starting NVD Severity Classifier Web Application...")
    print("📁 Checking for model artifacts...")
    
    # Check if artifacts exist
    artifacts_exist = os.path.exists('artifacts/model_training/xgb_severity_model.pkl')
    print(f"✅ Model file exists: {artifacts_exist}")
    
    if not artifacts_exist:
        print("❌ WARNING: Model artifacts not found. Please ensure your model files are in the correct location.")
        print("📁 Expected model path: artifacts/model_training/xgb_severity_model.pkl")
    
    print("🌐 Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)