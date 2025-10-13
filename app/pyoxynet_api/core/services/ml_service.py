"""
ML Inference Service for CPET Analysis
Uses TensorFlow Lite for lightweight inference without full TensorFlow dependency
"""
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        raise ImportError("TensorFlow Lite runtime not available")


class CPETMLService:
    """
    ML Service for CPET data analysis using TensorFlow Lite
    
    Handles:
    - TensorFlow Lite model loading
    - Data preprocessing for ML inference  
    - Exercise domain probability prediction
    - VT1/VT2 detection
    """
    
    def __init__(self, model_path: Optional[str] = None, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        try:
            self.model_path = model_path or self._resolve_model_path()
            self.interpreter = None
            self.input_details = None
            self.output_details = None
            
            self._load_tflite_model()
        except FileNotFoundError as e:
            self.logger.warning(f"TensorFlow Lite model not found: {e}")
            self.model_path = None
            self.interpreter = None
            self.input_details = None
            self.output_details = None
    
    def _resolve_model_path(self) -> str:
        """Resolve TensorFlow Lite model path"""
        possible_paths = [
            "tf_lite_models/tfl_model.tflite",
            "../tf_lite_models/tfl_model.tflite",
            str(Path(__file__).parent.parent.parent.parent / "tf_lite_models" / "tfl_model.tflite"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                self.logger.info(f"Found TensorFlow Lite model at: {path}")
                return path
        
        raise FileNotFoundError(f"TensorFlow Lite model not found in: {possible_paths}")
    
    def _load_tflite_model(self) -> None:
        """Load TensorFlow Lite model"""
        try:
            self.interpreter = tflite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            self.logger.info("TensorFlow Lite model loaded successfully")
            self.logger.debug(f"Input shape: {self.input_details[0]['shape']}")
            self.logger.debug(f"Output shape: {self.output_details[0]['shape']}")
            
        except Exception as e:
            self.logger.error(f"Failed to load TensorFlow Lite model: {e}")
            raise
    
    def predict_exercise_domains(self, data: pd.DataFrame) -> Dict:
        """
        Predict exercise domain probabilities for CPET data using moving window analysis
        
        Args:
            data: DataFrame with CPET measurements
            
        Returns:
            Dictionary with domain predictions, probabilities, and threshold detection
        """
        try:
            if self.interpreter is None:
                raise ValueError("TensorFlow Lite model not loaded. Cannot perform inference.")
            
            # Validate minimum data requirements
            if len(data) < 40:
                raise ValueError(f"Insufficient data: {len(data)} points. Minimum 40 points required for pyoxynet analysis.")
            
            # Run moving window analysis for complete CPET test
            moving_window_results = self._moving_window_analysis(data)
            
            # Detect ventilatory thresholds from moving window predictions
            threshold_results = self._detect_thresholds(moving_window_results, data)
            
            # Aggregate results
            domain_results = self._aggregate_window_results(moving_window_results, threshold_results, data)
            
            self.logger.info(f"Moving window analysis completed: {len(moving_window_results['time_predictions'])} predictions")
            return domain_results
            
        except Exception as e:
            self.logger.error(f"Moving window analysis failed: {e}")
            raise
    
    def _preprocess_for_tflite(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess CPET data for TensorFlow Lite model"""
        try:
            # Expected input: [batch, time_steps, features] = [1, 40, 5]
            # Based on original app.py, model expects exactly these 5 variables:
            # ['VO2_I', 'VCO2_I', 'VE_I', 'PetO2_I', 'PetCO2_I']
            required_columns = ['VO2', 'VCO2', 'VE', 'PetO2', 'PetCO2']
            
            # Map common column variations to standard names
            column_mapping = {
                'VO2_I': 'VO2', 'VO2_F': 'VO2',
                'VCO2_I': 'VCO2', 'VCO2_F': 'VCO2', 
                'VE_I': 'VE', 'VE_F': 'VE',
                'PetO2_I': 'PetO2', 'PetO2_F': 'PetO2',
                'PetCO2_I': 'PetCO2', 'PetCO2_F': 'PetCO2'
            }
            
            # Apply column mapping
            data_mapped = data.copy()
            for old_col, new_col in column_mapping.items():
                if old_col in data_mapped.columns and new_col not in data_mapped.columns:
                    data_mapped[new_col] = data_mapped[old_col]
            
            # Check required columns
            if not all(col in data_mapped.columns for col in required_columns):
                missing = [col for col in required_columns if col not in data_mapped.columns]
                available = list(data_mapped.columns)
                raise ValueError(f"Missing required columns: {missing}. Available columns: {available}")
            
            # Extract the 5 features exactly as expected by the model
            features = data_mapped[required_columns].copy()
            
            # Fill missing values
            features = features.ffill().bfill()
            
            # Normalize data exactly like the original app.py:
            # Each column normalized to [0,1] using (col - col.min()) / (col.max() - col.min())
            features_normalized = features.copy()
            for col in required_columns:
                col_min = features[col].min()
                col_max = features[col].max()
                if col_max != col_min:  # Avoid division by zero
                    features_normalized[col] = (features[col] - col_min) / (col_max - col_min)
                else:
                    features_normalized[col] = 0.0
            
            # Convert to numpy array with exact column order: ['VO2', 'VCO2', 'VE', 'PetO2', 'PetCO2']
            features_array = features_normalized[required_columns].values
            
            # Create windowed input: Use a sliding window approach like the original
            # Original uses past_points (which appears to be 40 based on model input shape)
            target_length = 40
            
            if len(features_array) >= target_length:
                # Take the last 40 points for inference
                windowed_data = features_array[-target_length:]
            else:
                # If insufficient data, pad with the first available values
                padded_data = np.zeros((target_length, 5))
                # Fill the end with actual data
                padded_data[-len(features_array):] = features_array
                # Fill the beginning with the first row repeated
                if len(features_array) > 0:
                    for i in range(target_length - len(features_array)):
                        padded_data[i] = features_array[0]
                windowed_data = padded_data
            
            # Reshape for model input: [1, 40, 5]
            input_data = windowed_data.reshape(1, target_length, 5).astype(np.float32)
            
            self.logger.debug(f"Preprocessed data shape: {input_data.shape}")
            return input_data
            
        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {e}")
            raise
    
    def _moving_window_analysis(self, data: pd.DataFrame) -> Dict:
        """
        Perform moving window analysis across the entire CPET test
        
        Returns predictions for each time point using a 40-point sliding window
        """
        try:
            # Prepare data for moving window
            required_columns = ['VO2', 'VCO2', 'VE', 'PetO2', 'PetCO2']
            
            # Map column variations
            column_mapping = {
                'VO2_I': 'VO2', 'VO2_F': 'VO2',
                'VCO2_I': 'VCO2', 'VCO2_F': 'VCO2', 
                'VE_I': 'VE', 'VE_F': 'VE',
                'PetO2_I': 'PetO2', 'PetO2_F': 'PetO2',
                'PetCO2_I': 'PetCO2', 'PetCO2_F': 'PetCO2'
            }
            
            data_mapped = data.copy()
            for old_col, new_col in column_mapping.items():
                if old_col in data_mapped.columns and new_col not in data_mapped.columns:
                    data_mapped[new_col] = data_mapped[old_col]
            
            # Check required columns
            if not all(col in data_mapped.columns for col in required_columns):
                missing = [col for col in required_columns if col not in data_mapped.columns]
                raise ValueError(f"Missing required columns: {missing}")
            
            # Get features and normalize
            features = data_mapped[required_columns].copy()
            features = features.ffill().bfill()
            
            # Normalize each column
            features_normalized = features.copy()
            for col in required_columns:
                col_min = features[col].min()
                col_max = features[col].max()
                if col_max != col_min:
                    features_normalized[col] = (features[col] - col_min) / (col_max - col_min)
                else:
                    features_normalized[col] = 0.0
            
            features_array = features_normalized[required_columns].values
            
            # Sliding window predictions
            window_size = 40
            predictions_over_time = []
            time_indices = []
            
            # Start predictions from point 40 onwards (when we have a full window)
            for i in range(window_size - 1, len(features_array)):
                # Extract 40-point window ending at current time point
                window_data = features_array[i - window_size + 1:i + 1]
                
                # Reshape for model input: [1, 40, 5]
                input_data = window_data.reshape(1, window_size, 5).astype(np.float32)
                
                # Run inference
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()
                prediction = self.interpreter.get_tensor(self.output_details[0]['index'])
                
                # Convert to probabilities
                prediction = prediction.flatten()
                exp_pred = np.exp(prediction - np.max(prediction))
                probabilities = exp_pred / np.sum(exp_pred)
                
                predictions_over_time.append(probabilities)
                time_indices.append(i)
            
            # Extract time values if available
            time_values = None
            time_cols = [col for col in data.columns if col.lower() in ['time', 't']]
            if time_cols:
                time_col = time_cols[0]
                time_values = data[time_col].iloc[time_indices].values
            
            return {
                'time_predictions': predictions_over_time,
                'time_indices': time_indices,
                'time_values': time_values,
                'window_size': window_size,
                'domain_names': ['Moderate', 'Heavy', 'Severe']
            }
            
        except Exception as e:
            self.logger.error(f"Moving window analysis failed: {e}")
            raise
    
    def _detect_thresholds(self, moving_results: Dict, data: pd.DataFrame) -> Dict:
        """
        Detect VT1 and VT2 from moving window predictions
        
        VT1: Transition from Moderate to Heavy domain
        VT2: Transition from Heavy to Severe domain
        """
        try:
            predictions = np.array(moving_results['time_predictions'])
            time_indices = moving_results['time_indices']
            
            # Get dominant domain at each time point
            dominant_domains = np.argmax(predictions, axis=1)  # 0=Moderate, 1=Heavy, 2=Severe
            
            # Smooth predictions to reduce noise (simple moving average)
            def simple_smooth(data, window=5):
                """Simple moving average smoothing"""
                smoothed = np.copy(data)
                half_window = window // 2
                for i in range(half_window, len(data) - half_window):
                    smoothed[i] = np.mean(data[i-half_window:i+half_window+1])
                return smoothed
            
            smoothed_moderate = simple_smooth(predictions[:, 0])
            smoothed_heavy = simple_smooth(predictions[:, 1])
            smoothed_severe = simple_smooth(predictions[:, 2])
            
            # Find transitions
            vt1_index = None
            vt2_index = None
            
            # VT1: First significant transition from Moderate dominance
            for i in range(10, len(smoothed_moderate) - 10):
                # Look for sustained decrease in moderate domain probability
                before_moderate = np.mean(smoothed_moderate[i-10:i])
                after_moderate = np.mean(smoothed_moderate[i:i+10])
                
                if before_moderate > 0.6 and after_moderate < 0.4 and smoothed_heavy[i] > 0.3:
                    vt1_index = time_indices[i]
                    break
            
            # VT2: Transition to Severe domain dominance
            for i in range(10, len(smoothed_severe) - 10):
                # Look for sustained increase in severe domain probability
                before_severe = np.mean(smoothed_severe[i-10:i])
                after_severe = np.mean(smoothed_severe[i:i+10])
                
                if before_severe < 0.3 and after_severe > 0.5:
                    vt2_index = time_indices[i]
                    break
            
            # Convert indices to VO2 values
            vt1_vo2 = None
            vt2_vo2 = None
            
            if 'VO2' in data.columns:
                if vt1_index is not None and vt1_index < len(data):
                    vt1_vo2 = float(data['VO2'].iloc[vt1_index])
                if vt2_index is not None and vt2_index < len(data):
                    vt2_vo2 = float(data['VO2'].iloc[vt2_index])
            
            return {
                'VT1': {
                    'index': vt1_index,
                    'vo2': vt1_vo2,
                    'detected': vt1_vo2 is not None
                },
                'VT2': {
                    'index': vt2_index,
                    'vo2': vt2_vo2,
                    'detected': vt2_vo2 is not None
                },
                'dominant_domains_over_time': dominant_domains.tolist()
            }
            
        except Exception as e:
            self.logger.warning(f"Threshold detection failed: {e}")
            # Fallback to simple estimation
            vt1_estimate = None
            vt2_estimate = None
            if 'VO2' in data.columns:
                vo2_max = data['VO2'].max()
                vt1_estimate = float(vo2_max * 0.6)
                vt2_estimate = float(vo2_max * 0.8)
            
            return {
                'VT1': {'vo2': vt1_estimate, 'detected': False},
                'VT2': {'vo2': vt2_estimate, 'detected': False},
                'dominant_domains_over_time': []
            }
    
    def _aggregate_window_results(self, moving_results: Dict, threshold_results: Dict, data: pd.DataFrame) -> Dict:
        """
        Aggregate moving window results into final analysis
        """
        try:
            predictions = np.array(moving_results['time_predictions'])
            
            # Calculate overall domain probabilities (mean across time)
            mean_probabilities = np.mean(predictions, axis=0)
            domain_names = moving_results['domain_names']
            
            domain_probabilities = {
                domain: float(prob) for domain, prob in zip(domain_names, mean_probabilities)
            }
            
            # Dominant domain
            dominant_idx = np.argmax(mean_probabilities)
            dominant_domain = domain_names[dominant_idx]
            confidence = float(mean_probabilities[dominant_idx])
            
            # Extract threshold values
            vt1_vo2 = threshold_results['VT1']['vo2']
            vt2_vo2 = threshold_results['VT2']['vo2']
            
            results = {
                'domain_probabilities': domain_probabilities,
                'dominant_domain': dominant_domain,
                'confidence': confidence,
                'ventilatory_thresholds': {
                    'VT1': vt1_vo2,
                    'VT2': vt2_vo2
                },
                'threshold_detection': {
                    'VT1_detected': threshold_results['VT1']['detected'],
                    'VT2_detected': threshold_results['VT2']['detected'],
                    'VT1_index': threshold_results['VT1'].get('index'),
                    'VT2_index': threshold_results['VT2'].get('index')
                },
                'temporal_analysis': {
                    'predictions_over_time': [pred.tolist() for pred in predictions],
                    'time_points': len(predictions),
                    'dominant_domains_over_time': threshold_results['dominant_domains_over_time'],
                    'time_indices': moving_results['time_indices']
                },
                'data_quality': self._assess_data_quality(data),
                'moving_window_info': {
                    'window_size': moving_results['window_size'],
                    'total_windows': len(predictions),
                    'analysis_coverage': f"{len(predictions)}/{len(data)} points"
                }
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Result aggregation failed: {e}")
            raise
    
    def _process_tflite_predictions(self, predictions: np.ndarray, original_data: pd.DataFrame) -> Dict:
        """
        Process TensorFlow Lite predictions into standardized API response format
        
        Args:
            predictions: Raw TFLite model outputs
            original_data: Original CPET data for context
            
        Returns:
            Standardized domain analysis results
        """
        try:
            # TFLite model typically outputs [batch, classes] = [1, 3] for [Moderate, Heavy, Severe]
            predictions = predictions.flatten()  # Convert to 1D array
            
            # Apply softmax to get probabilities
            exp_preds = np.exp(predictions - np.max(predictions))  # Numerical stability
            probabilities = exp_preds / np.sum(exp_preds)
            
            domain_names = ['Moderate', 'Heavy', 'Severe']
            domain_probabilities = {
                domain: float(prob) for domain, prob in zip(domain_names, probabilities)
            }
            
            # Find dominant domain
            dominant_idx = np.argmax(probabilities)
            dominant_domain = domain_names[dominant_idx]
            confidence = float(probabilities[dominant_idx])
            
            # Simple VT estimation based on VO2 data (placeholder logic)
            vt1_estimate = None
            vt2_estimate = None
            if 'VO2' in original_data.columns:
                vo2_values = original_data['VO2'].values
                if len(vo2_values) > 10:
                    # Simple threshold estimation: VT1 at ~60% of max VO2, VT2 at ~80%
                    vo2_max = np.max(vo2_values)
                    vt1_estimate = float(vo2_max * 0.6)
                    vt2_estimate = float(vo2_max * 0.8)
            
            results = {
                'domain_probabilities': domain_probabilities,
                'dominant_domain': dominant_domain,
                'confidence': confidence,
                'ventilatory_thresholds': {
                    'VT1': vt1_estimate,
                    'VT2': vt2_estimate
                },
                'temporal_analysis': {
                    'predictions_over_time': [probabilities.tolist()],  # Single prediction
                    'time_points': 1
                },
                'data_quality': self._assess_data_quality(original_data),
                'tflite_raw_predictions': {
                    'raw_output': predictions.tolist(),
                    'probabilities': probabilities.tolist(),
                    'model_input_shape': self.input_details[0]['shape'].tolist() if self.input_details else None,
                    'model_output_shape': self.output_details[0]['shape'].tolist() if self.output_details else None
                }
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"TFLite result processing failed: {e}")
            raise
    
    def _assess_data_quality(self, data: pd.DataFrame) -> Dict:
        """Assess quality of CPET data for analysis"""
        try:
            quality_metrics = {
                'total_points': len(data),
                'missing_values': data.isnull().sum().to_dict(),
                'data_range': {},
                'duration_minutes': 0,
            }
            
            # Add data ranges for key columns
            for col in ['VO2', 'VCO2', 'VE', 'HR']:
                if col in data.columns:
                    quality_metrics['data_range'][col] = {
                        'min': float(data[col].min()), 
                        'max': float(data[col].max())
                    }
            
            # Estimate duration if time column exists
            time_cols = [col for col in data.columns if col.lower() in ['time', 't']]
            if time_cols:
                time_col = time_cols[0]
                time_vals = data[time_col].dropna()
                if len(time_vals) > 1:
                    quality_metrics['duration_minutes'] = float((time_vals.max() - time_vals.min()) / 60)
            
            # Overall quality score
            quality_score = 1.0
            if len(data) < 100:  # Less than ~2 minutes of data
                quality_score *= 0.7
            if data.isnull().sum().sum() > len(data) * 0.1:  # More than 10% missing
                quality_score *= 0.8
                
            quality_metrics['quality_score'] = quality_score
            quality_metrics['quality_rating'] = (
                'Excellent' if quality_score > 0.9 else
                'Good' if quality_score > 0.7 else
                'Fair' if quality_score > 0.5 else 'Poor'
            )
            
            return quality_metrics
            
        except Exception as e:
            self.logger.warning(f"Data quality assessment failed: {e}")
            return {
                'total_points': len(data),
                'quality_score': 0.5,
                'quality_rating': 'Unknown'
            }
    
    def get_model_info(self) -> Dict:
        """Get model information - compatibility method"""
        health_info = self.health_check()
        return health_info.get("model_info", {})
    
    def health_check(self) -> Dict:
        """Check if the ML service is healthy"""
        try:
            status = "ready" if self.interpreter is not None else "error"
            
            # Ensure all arrays are converted to lists for JSON serialization
            input_shape = None
            output_shape = None
            if self.input_details:
                input_shape = [int(x) for x in self.input_details[0]['shape']]
            if self.output_details:
                output_shape = [int(x) for x in self.output_details[0]['shape']]
            
            return {
                "status": status,
                "model_info": {
                    "model_path": str(self.model_path) if self.model_path else None,
                    "model_loaded": self.interpreter is not None,
                    "input_shape": input_shape,
                    "output_shape": output_shape,
                    "tflite_runtime_available": True
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "model_info": {
                    "model_path": str(getattr(self, 'model_path', None)) if hasattr(self, 'model_path') else None,
                    "model_loaded": False,
                    "tflite_runtime_available": False,
                    "error": str(e)
                }
            }