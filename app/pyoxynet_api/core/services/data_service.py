"""
Data Processing Service for CPET Analysis
Handles file parsing, data preprocessing, and validation
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import io
from werkzeug.datastructures import FileStorage


class CPETDataService:
    """
    Service for CPET data processing and file handling
    
    Handles:
    - Multi-format file parsing (CSV, Excel, TXT)
    - Metabolimeter-specific data formats
    - Data preprocessing and cleaning
    - Column standardization and validation
    """
    
    # Standard CPET column mappings for different metabolimeters
    COLUMN_MAPPINGS = {
        'generic': {
            'time': ['Time', 'time', 't', 'T'],
            'vo2': ['VO2', 'vo2', 'O2', 'VO2(ml/min)', 'VO2 (ml/min)'],
            'vco2': ['VCO2', 'vco2', 'CO2', 'VCO2(ml/min)', 'VCO2 (ml/min)'],
            've': ['VE', 've', 'VE(l/min)', 'VE (l/min)', 'Ventilation'],
            'hr': ['HR', 'hr', 'Heart Rate', 'HeartRate', 'BPM'],
            'rer': ['RER', 'rer', 'RQ', 'rq', 'R'],
            'load': ['Load', 'load', 'Work', 'work', 'Watts', 'watts', 'W']
        },
        'cortex': {
            'time': ['t', 'Time'],
            'vo2': ['VO2', 'VO2/kg'],
            'vco2': ['VCO2'],
            've': ['VE'],
            'hr': ['HR'],
            'rer': ['RER'],
            'load': ['Load']
        },
        'cosmed': {
            'time': ['Time'],
            'vo2': ['VO2', 'VO2/kg'],
            'vco2': ['VCO2'],
            've': ['VE'],
            'hr': ['HR'],
            'rer': ['R'],
            'load': ['WR']
        }
    }
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def process_file(self, file_storage: FileStorage, session_id: str) -> Dict:
        """
        Process uploaded file and return parsed CPET data
        
        Args:
            file_storage: Uploaded file from Flask
            session_id: User session identifier
            
        Returns:
            Dictionary with processed data and metadata
        """
        try:
            # Determine file type and parse
            filename = file_storage.filename.lower()
            
            if filename.endswith('.csv'):
                data = self._parse_csv(file_storage)
            elif filename.endswith(('.xlsx', '.xls')):
                data = self._parse_excel(file_storage)
            elif filename.endswith('.txt'):
                data = self._parse_txt(file_storage)
            else:
                raise ValueError(f"Unsupported file type: {filename}")
            
            # Detect metabolimeter type
            metabolimeter_type = self._detect_metabolimeter(data)
            
            # Standardize column names
            standardized_data = self._standardize_columns(data, metabolimeter_type)
            
            # Validate and clean data
            cleaned_data = self._clean_data(standardized_data)
            
            # Generate metadata
            metadata = self._generate_metadata(cleaned_data, filename, metabolimeter_type)
            
            self.logger.info(f"Successfully processed file: {filename} ({len(cleaned_data)} rows)")
            
            return {
                'data': cleaned_data,
                'metadata': metadata,
                'success': True,
                'errors': []
            }
            
        except Exception as e:
            self.logger.error(f"File processing failed: {e}")
            return {
                'data': None,
                'metadata': None,
                'success': False,
                'errors': [str(e)]
            }
    
    def _parse_csv(self, file_storage: FileStorage) -> pd.DataFrame:
        """Parse CSV file with flexible delimiter detection"""
        try:
            # Read file content
            content = file_storage.read().decode('utf-8')
            file_storage.seek(0)  # Reset for potential re-read
            
            # Try different delimiters
            delimiters = [',', ';', '\t']
            
            for delimiter in delimiters:
                try:
                    data = pd.read_csv(io.StringIO(content), delimiter=delimiter)
                    if len(data.columns) > 1:  # Successful parsing
                        self.logger.debug(f"CSV parsed with delimiter: '{delimiter}'")
                        return data
                except Exception:
                    continue
            
            # Fallback to pandas auto-detection
            data = pd.read_csv(io.StringIO(content))
            return data
            
        except Exception as e:
            raise ValueError(f"Failed to parse CSV file: {e}")
    
    def _parse_excel(self, file_storage: FileStorage) -> pd.DataFrame:
        """Parse Excel file"""
        try:
            # Try reading the first sheet
            data = pd.read_excel(file_storage, engine='openpyxl')
            return data
            
        except Exception as e:
            raise ValueError(f"Failed to parse Excel file: {e}")
    
    def _parse_txt(self, file_storage: FileStorage) -> pd.DataFrame:
        """Parse TXT file (usually tab-delimited)"""
        try:
            content = file_storage.read().decode('utf-8')
            
            # Try tab-delimited first
            try:
                data = pd.read_csv(io.StringIO(content), delimiter='\t')
                if len(data.columns) > 1:
                    return data
            except Exception:
                pass
            
            # Try space-delimited
            try:
                data = pd.read_csv(io.StringIO(content), delimiter=' ', skipinitialspace=True)
                return data
            except Exception as e:
                raise ValueError(f"Failed to parse TXT file: {e}")
                
        except Exception as e:
            raise ValueError(f"Failed to read TXT file: {e}")
    
    def _detect_metabolimeter(self, data: pd.DataFrame) -> str:
        """Detect metabolimeter type based on column patterns"""
        columns = [col.lower() for col in data.columns]
        
        # Check for specific patterns
        if 'rer' in columns and any('vo2/kg' in col.lower() for col in data.columns):
            return 'cortex'
        elif 'r' in columns and 'wr' in columns:
            return 'cosmed'
        else:
            return 'generic'
    
    def _standardize_columns(self, data: pd.DataFrame, metabolimeter_type: str) -> pd.DataFrame:
        """
        Standardize column names to common CPET variables
        
        Args:
            data: Raw data DataFrame
            metabolimeter_type: Detected metabolimeter type
            
        Returns:
            DataFrame with standardized column names
        """
        try:
            mappings = self.COLUMN_MAPPINGS.get(metabolimeter_type, self.COLUMN_MAPPINGS['generic'])
            standardized_data = data.copy()
            
            # Create reverse mapping for renaming
            rename_map = {}
            
            for standard_name, possible_names in mappings.items():
                for original_col in data.columns:
                    if original_col in possible_names or original_col.lower() in [n.lower() for n in possible_names]:
                        rename_map[original_col] = standard_name.upper()
                        break
            
            # Rename columns
            standardized_data = standardized_data.rename(columns=rename_map)
            
            self.logger.debug(f"Column mapping applied: {rename_map}")
            
            return standardized_data
            
        except Exception as e:
            self.logger.warning(f"Column standardization failed: {e}")
            return data
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate CPET data
        
        Args:
            data: DataFrame with standardized columns
            
        Returns:
            Cleaned DataFrame
        """
        try:
            cleaned = data.copy()
            
            # Remove completely empty rows
            cleaned = cleaned.dropna(how='all')
            
            # Ensure numeric columns are numeric
            numeric_columns = ['VO2', 'VCO2', 'VE', 'HR', 'RER', 'LOAD']
            for col in numeric_columns:
                if col in cleaned.columns:
                    cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')
            
            # Handle time column
            if 'TIME' in cleaned.columns:
                cleaned['TIME'] = pd.to_numeric(cleaned['TIME'], errors='coerce')
                cleaned = cleaned.sort_values('TIME')
            elif len(cleaned) > 0:
                # Create time index if missing
                cleaned['TIME'] = np.arange(len(cleaned))
            
            # Remove invalid data points
            if 'VO2' in cleaned.columns:
                # Remove negative or extremely high VO2 values
                valid_vo2 = (cleaned['VO2'] >= 0) & (cleaned['VO2'] <= 10000)  # ml/min
                cleaned = cleaned[valid_vo2]
            
            if 'VCO2' in cleaned.columns:
                # Remove negative VCO2 values
                valid_vco2 = cleaned['VCO2'] >= 0
                cleaned = cleaned[valid_vco2]
            
            if 'VE' in cleaned.columns:
                # Remove negative VE values
                valid_ve = cleaned['VE'] >= 0
                cleaned = cleaned[valid_ve]
            
            # Calculate derived variables if possible
            cleaned = self._calculate_derived_variables(cleaned)
            
            # Reset index
            cleaned = cleaned.reset_index(drop=True)
            
            self.logger.debug(f"Data cleaned: {len(data)} -> {len(cleaned)} rows")
            
            return cleaned
            
        except Exception as e:
            self.logger.error(f"Data cleaning failed: {e}")
            return data
    
    def _calculate_derived_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived CPET variables"""
        try:
            result = data.copy()
            
            # Calculate RER if not present
            if 'RER' not in result.columns and 'VO2' in result.columns and 'VCO2' in result.columns:
                result['RER'] = result['VCO2'] / result['VO2']
                result['RER'] = result['RER'].clip(0.5, 2.0)  # Physiological limits
            
            # Calculate VE/VO2 and VE/VCO2 ratios
            if 'VE' in result.columns and 'VO2' in result.columns:
                result['VE_VO2'] = (result['VE'] * 1000) / result['VO2']  # Convert L to mL
            
            if 'VE' in result.columns and 'VCO2' in result.columns:
                result['VE_VCO2'] = (result['VE'] * 1000) / result['VCO2']  # Convert L to mL
            
            # Calculate O2 pulse if HR is available
            if 'VO2' in result.columns and 'HR' in result.columns:
                result['O2_PULSE'] = result['VO2'] / result['HR']
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Derived variable calculation failed: {e}")
            return data
    
    def _generate_metadata(self, data: pd.DataFrame, filename: str, metabolimeter_type: str) -> Dict:
        """Generate metadata about the processed data"""
        try:
            # Basic statistics
            metadata = {
                'filename': filename,
                'metabolimeter_type': metabolimeter_type,
                'total_rows': len(data),
                'columns': list(data.columns),
                'duration_minutes': None,
                'sampling_rate': None,
                'data_quality': 'Good',
                'available_variables': {
                    'VO2': 'VO2' in data.columns,
                    'VCO2': 'VCO2' in data.columns,
                    'VE': 'VE' in data.columns,
                    'HR': 'HR' in data.columns,
                    'RER': 'RER' in data.columns,
                    'LOAD': 'LOAD' in data.columns
                }
            }
            
            # Time-related metadata
            if 'TIME' in data.columns and len(data) > 1:
                time_diff = data['TIME'].iloc[-1] - data['TIME'].iloc[0]
                metadata['duration_minutes'] = float(time_diff / 60)
                metadata['sampling_rate'] = float(len(data) / (time_diff / 60))  # Hz
            
            # Variable ranges
            variable_ranges = {}
            for var in ['VO2', 'VCO2', 'VE', 'HR', 'RER']:
                if var in data.columns:
                    variable_ranges[var] = {
                        'min': float(data[var].min()),
                        'max': float(data[var].max()),
                        'mean': float(data[var].mean()),
                        'std': float(data[var].std())
                    }
            
            metadata['variable_ranges'] = variable_ranges
            
            # Data quality assessment
            missing_percentage = data.isnull().sum().sum() / (len(data) * len(data.columns))
            if missing_percentage > 0.2:
                metadata['data_quality'] = 'Poor'
            elif missing_percentage > 0.1:
                metadata['data_quality'] = 'Fair'
            
            metadata['missing_data_percentage'] = float(missing_percentage)
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Metadata generation failed: {e}")
            return {
                'filename': filename,
                'metabolimeter_type': metabolimeter_type,
                'total_rows': len(data) if data is not None else 0,
                'data_quality': 'Unknown'
            }
    
    def validate_cpet_data(self, data: pd.DataFrame) -> Dict:
        """
        Validate CPET data for analysis readiness
        
        Args:
            data: Processed CPET DataFrame
            
        Returns:
            Validation results with errors and warnings
        """
        errors = []
        warnings = []
        
        try:
            # Check minimum required columns
            required_columns = ['VO2', 'VCO2', 'VE']
            missing_required = [col for col in required_columns if col not in data.columns]
            
            if missing_required:
                errors.append(f"Missing required columns: {missing_required}")
            
            # Check data length
            if len(data) < 30:
                warnings.append("Very short test duration (< 30 data points)")
            elif len(data) < 60:
                warnings.append("Short test duration (< 60 data points)")
            
            # Check for reasonable value ranges
            if 'VO2' in data.columns:
                vo2_max = data['VO2'].max()
                if vo2_max < 500:
                    warnings.append("Unusually low maximum VO2 values")
                elif vo2_max > 8000:
                    warnings.append("Unusually high maximum VO2 values")
            
            # Check for data gaps
            if 'TIME' in data.columns:
                time_gaps = data['TIME'].diff().fillna(0)
                large_gaps = time_gaps > time_gaps.median() * 3
                if large_gaps.sum() > 0:
                    warnings.append(f"Found {large_gaps.sum()} large time gaps in data")
            
            # Overall validation result
            is_valid = len(errors) == 0
            
            return {
                'valid': is_valid,
                'errors': errors,
                'warnings': warnings,
                'recommendations': self._get_recommendations(errors, warnings)
            }
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            return {
                'valid': False,
                'errors': [f"Validation failed: {str(e)}"],
                'warnings': [],
                'recommendations': []
            }
    
    def _get_recommendations(self, errors: List[str], warnings: List[str]) -> List[str]:
        """Generate recommendations based on validation errors and warnings"""
        recommendations = []
        
        if errors:
            recommendations.append("Address validation errors before proceeding with analysis")
        
        if warnings:
            if any("short" in w.lower() for w in warnings):
                recommendations.append("Consider longer test duration for more reliable results")
            
            if any("gap" in w.lower() for w in warnings):
                recommendations.append("Check data acquisition settings to avoid time gaps")
            
            if any("vo2" in w.lower() for w in warnings):
                recommendations.append("Verify VO2 sensor calibration and measurement units")
        
        return recommendations