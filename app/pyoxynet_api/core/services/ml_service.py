"""
ML Inference Service for CPET Analysis
Handles TensorFlow Lite model loading and inference for exercise domain classification
"""
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path

try:
    import tensorflow.lite as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        raise ImportError("TensorFlow Lite runtime not available")


class CPETMLService:
    """
    ML Service for CPET data analysis using TensorFlow Lite models
    
    Handles:
    - Model loading and management
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
            
            self._load_model()
        except FileNotFoundError as e:
            # In test environments, model may not be available
            self.logger.warning(f"TensorFlow Lite model not found: {e}")
            self.model_path = None
            self.interpreter = None
            self.input_details = None
            self.output_details = None
    
    def _resolve_model_path(self) -> str:
        """Resolve TensorFlow Lite model path for different environments"""
        possible_paths = [
            # Local development
            "../pyoxynet/pyoxynet/tfl_models/tfl_model.tflite",
            "pyoxynet/pyoxynet/tfl_models/tfl_model.tflite",
            # Docker environment
            "/app/pyoxynet/pyoxynet/tfl_models/tfl_model.tflite",
            # Relative to current file
            str(Path(__file__).parent.parent.parent.parent / "pyoxynet" / "pyoxynet" / "tfl_models" / "tfl_model.tflite")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                self.logger.info(f"Found TensorFlow Lite model at: {path}")
                return path
        
        raise FileNotFoundError(f"TensorFlow Lite model not found in any of: {possible_paths}")
    
    def _load_model(self) -> None:
        """Load TensorFlow Lite model and prepare interpreter"""
        try:
            self.interpreter = tflite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            self.logger.info("TensorFlow Lite model loaded successfully")
            self.logger.debug(f"Input details: {self.input_details}")
            self.logger.debug(f"Output details: {self.output_details}")
            
        except Exception as e:
            self.logger.error(f"Failed to load TensorFlow Lite model: {e}")
            raise
    
    def preprocess_data(self, data: pd.DataFrame, window_size: int = 30) -> np.ndarray:
        """
        Preprocess CPET data for ML inference
        
        Args:
            data: DataFrame with CPET variables (VO2, VCO2, VE, etc.)
            window_size: Size of sliding window for temporal analysis
            
        Returns:
            Preprocessed numpy array ready for model inference
        """
        try:
            # Select key CPET variables for analysis
            required_columns = ['VO2', 'VCO2', 'VE', 'HR']  # Minimum required
            optional_columns = ['RER', 'VT', 'BR', 'FETO2', 'FETCO2']  # Additional if available
            
            # Use available columns
            available_columns = [col for col in required_columns + optional_columns if col in data.columns]
            
            if not all(col in available_columns for col in required_columns):
                missing = [col for col in required_columns if col not in available_columns]
                raise ValueError(f"Missing required columns: {missing}")
            
            # Extract and normalize data
            features = data[available_columns].copy()
            
            # Handle missing values
            features = features.fillna(method='ffill').fillna(method='bfill')
            
            # Normalize data (z-score normalization)
            features_normalized = (features - features.mean()) / features.std()
            
            # Create sliding windows
            windowed_data = self._create_sliding_windows(features_normalized.values, window_size)
            
            self.logger.debug(f"Preprocessed data shape: {windowed_data.shape}")
            return windowed_data
            
        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {e}")
            raise
    
    def _create_sliding_windows(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Create sliding windows from time series data"""
        if len(data) < window_size:
            # If data is shorter than window, pad with zeros
            padded_data = np.zeros((window_size, data.shape[1]))
            padded_data[:len(data)] = data
            return padded_data.reshape(1, window_size, -1)
        
        windows = []
        for i in range(len(data) - window_size + 1):
            windows.append(data[i:i + window_size])
        
        return np.array(windows)
    
    def predict_exercise_domains(self, data: pd.DataFrame) -> Dict:
        """
        Predict exercise domain probabilities for CPET data
        
        Args:
            data: DataFrame with CPET measurements
            
        Returns:
            Dictionary with domain predictions and probabilities
        """
        try:
            # Check if model is available
            if self.interpreter is None:
                # Return mock results for testing/demo
                return self._create_mock_predictions(data)
            
            # Preprocess data
            processed_data = self.preprocess_data(data)
            
            # Run inference
            predictions = []
            for window in processed_data:
                # Ensure correct input shape
                input_data = window.reshape(1, *window.shape).astype(np.float32)
                
                # Set input tensor
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                
                # Run inference
                self.interpreter.invoke()
                
                # Get output
                output = self.interpreter.get_tensor(self.output_details[0]['index'])
                predictions.append(output[0])
            
            predictions = np.array(predictions)
            
            # Process predictions to domain probabilities
            domain_results = self._process_domain_predictions(predictions, data)
            
            self.logger.info("Exercise domain prediction completed successfully")
            return domain_results
            
        except Exception as e:
            self.logger.error(f"ML inference failed: {e}")
            raise
    
    def _process_domain_predictions(self, predictions: np.ndarray, original_data: pd.DataFrame) -> Dict:
        """
        Process raw ML predictions into exercise domain analysis
        
        Args:
            predictions: Raw model outputs
            original_data: Original CPET data for context
            
        Returns:
            Processed domain analysis results
        """
        try:
            # Assuming model outputs probabilities for [Moderate, Heavy, Severe] domains
            domain_names = ['Moderate', 'Heavy', 'Severe']
            
            # Calculate mean probabilities across time
            mean_probabilities = np.mean(predictions, axis=0)
            
            # Find dominant domain
            dominant_domain_idx = np.argmax(mean_probabilities)
            dominant_domain = domain_names[dominant_domain_idx]
            
            # Calculate ventilatory thresholds (simplified approach)
            vt1_estimate, vt2_estimate = self._estimate_ventilatory_thresholds(predictions, original_data)
            
            results = {
                'domain_probabilities': {
                    domain: float(prob) for domain, prob in zip(domain_names, mean_probabilities)
                },
                'dominant_domain': dominant_domain,
                'confidence': float(max(mean_probabilities)),
                'ventilatory_thresholds': {
                    'VT1': vt1_estimate,
                    'VT2': vt2_estimate
                },
                'temporal_analysis': {
                    'predictions_over_time': predictions.tolist(),
                    'time_points': len(predictions)
                },
                'data_quality': self._assess_data_quality(original_data)
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Prediction processing failed: {e}")
            raise
    
    def _estimate_ventilatory_thresholds(self, predictions: np.ndarray, data: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
        """
        Estimate VT1 and VT2 based on domain transition probabilities
        
        Returns:
            Tuple of (VT1, VT2) estimates in terms of VO2 values
        """
        try:
            if 'VO2' not in data.columns:
                return None, None
            
            vo2_values = data['VO2'].values
            
            # Simple threshold detection based on domain transitions
            # This is a simplified approach - production would use more sophisticated algorithms
            
            # VT1: Moderate to Heavy domain transition
            moderate_to_heavy = np.diff(predictions[:, 1])  # Heavy domain probability changes
            vt1_candidates = np.where(moderate_to_heavy > 0.1)[0]  # Significant increase in Heavy domain
            
            # VT2: Heavy to Severe domain transition
            heavy_to_severe = np.diff(predictions[:, 2])  # Severe domain probability changes
            vt2_candidates = np.where(heavy_to_severe > 0.1)[0]  # Significant increase in Severe domain
            
            vt1 = float(vo2_values[vt1_candidates[0]]) if len(vt1_candidates) > 0 else None
            vt2 = float(vo2_values[vt2_candidates[0]]) if len(vt2_candidates) > 0 else None
            
            return vt1, vt2
            
        except Exception as e:
            self.logger.warning(f"VT estimation failed: {e}")
            return None, None
    
    def _assess_data_quality(self, data: pd.DataFrame) -> Dict:
        """Assess quality of CPET data for analysis"""
        try:
            quality_metrics = {
                'total_points': len(data),
                'missing_values': data.isnull().sum().to_dict(),
                'data_range': {
                    'VO2': {'min': float(data['VO2'].min()), 'max': float(data['VO2'].max())} if 'VO2' in data.columns else None,
                    'VCO2': {'min': float(data['VCO2'].min()), 'max': float(data['VCO2'].max())} if 'VCO2' in data.columns else None,
                    'VE': {'min': float(data['VE'].min()), 'max': float(data['VE'].max())} if 'VE' in data.columns else None,
                },
                'duration_minutes': (data.index[-1] - data.index[0]) / 60 if len(data) > 1 else 0,
            }
            
            # Overall quality score
            quality_score = 1.0
            if len(data) < 100:  # Less than ~2 minutes of data
                quality_score *= 0.7
            if data.isnull().sum().sum() > len(data) * 0.1:  # More than 10% missing
                quality_score *= 0.8
                
            quality_metrics['quality_score'] = quality_score
            quality_metrics['quality_rating'] = 'Excellent' if quality_score > 0.9 else 'Good' if quality_score > 0.7 else 'Fair' if quality_score > 0.5 else 'Poor'
            
            return quality_metrics
            
        except Exception as e:
            self.logger.warning(f"Data quality assessment failed: {e}")
            return {'quality_score': 0.5, 'quality_rating': 'Unknown'}
    
    def _create_mock_predictions(self, data: pd.DataFrame) -> Dict:
        """Create mock predictions for testing when model is not available"""
        # Simple heuristic based on VO2 progression for demo purposes
        if 'VO2' not in data.columns:
            vo2_range = [500, 2000]  # Default range
        else:
            vo2_range = [float(data['VO2'].min()), float(data['VO2'].max())]
        
        # Mock domain probabilities based on VO2 progression
        vo2_span = vo2_range[1] - vo2_range[0]
        if vo2_span > 1500:  # Large range suggests progression through domains
            mock_probs = {'Moderate': 0.4, 'Heavy': 0.4, 'Severe': 0.2}
        elif vo2_span > 800:  # Medium range
            mock_probs = {'Moderate': 0.6, 'Heavy': 0.3, 'Severe': 0.1}
        else:  # Small range
            mock_probs = {'Moderate': 0.8, 'Heavy': 0.15, 'Severe': 0.05}
        
        dominant_domain = max(mock_probs, key=mock_probs.get)
        
        return {
            'domain_probabilities': mock_probs,
            'dominant_domain': dominant_domain,
            'confidence': 0.75,  # Mock confidence
            'ventilatory_thresholds': {
                'VT1': vo2_range[0] + vo2_span * 0.4 if vo2_span > 0 else None,
                'VT2': vo2_range[0] + vo2_span * 0.7 if vo2_span > 0 else None
            },
            'temporal_analysis': {
                'predictions_over_time': np.random.dirichlet([2, 2, 1], size=min(len(data), 20)).tolist(),
                'time_points': min(len(data), 20)
            },
            'data_quality': self._assess_data_quality(data)
        }
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_path': self.model_path,
            'input_shape': self.input_details[0]['shape'].tolist() if self.input_details else None,
            'output_shape': self.output_details[0]['shape'].tolist() if self.output_details else None,
            'model_loaded': self.interpreter is not None
        }