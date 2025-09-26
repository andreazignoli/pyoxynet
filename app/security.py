"""
PyOxynet Security Module
Medical data privacy compliance and file validation for CPET analysis
"""
import os
import hashlib
import mimetypes
from typing import List, Dict, Any, Tuple, Optional
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
import pandas as pd
import tempfile
import threading
import time
from datetime import datetime, timedelta


class CPETFileValidator:
    """Validator for CPET data files with medical privacy compliance"""
    
    # Required CPET columns for scientific accuracy
    REQUIRED_COLUMNS = {'VO2', 'VCO2', 'VE'}  # Minimum required
    OPTIONAL_COLUMNS = {'HR', 'RF', 'PetO2', 'PetCO2', 'VEVO2', 'VEVCO2', 'Time'}
    ALLOWED_EXTENSIONS = {'csv', 'txt'}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB per file
    MAX_ROWS = 100000  # Maximum rows per CPET file
    
    def __init__(self, logger):
        self.logger = logger
    
    def validate_file(self, file: FileStorage) -> Dict[str, Any]:
        """
        Comprehensive file validation for CPET data
        Returns validation result with security and scientific checks
        """
        result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'file_info': {},
            'cpet_info': {}
        }
        
        try:
            # Basic file security validation
            security_result = self._validate_file_security(file)
            if not security_result['valid']:
                result['errors'].extend(security_result['errors'])
                return result
            
            result['file_info'] = security_result['file_info']
            
            # CPET-specific scientific validation
            cpet_result = self._validate_cpet_content(file)
            result['cpet_info'] = cpet_result['cpet_info']
            result['warnings'].extend(cpet_result['warnings'])
            
            if not cpet_result['valid']:
                result['errors'].extend(cpet_result['errors'])
                return result
            
            result['valid'] = True
            self.logger.log_security_event('file_validation_success', result['file_info'])
            
        except Exception as e:
            result['errors'].append(f"Validation error: {str(e)}")
            self.logger.log_security_event('file_validation_error', {'error': str(e)})
        
        return result
    
    def _validate_file_security(self, file: FileStorage) -> Dict[str, Any]:
        """Security validation for uploaded files"""
        result = {'valid': False, 'errors': [], 'file_info': {}}
        
        # Check file exists and has content
        if not file or not file.filename:
            result['errors'].append("No file provided")
            return result
        
        # Secure filename
        filename = secure_filename(file.filename)
        if not filename:
            result['errors'].append("Invalid filename")
            return result
        
        # File extension validation
        if not self._allowed_file(filename):
            result['errors'].append(f"File type not allowed. Allowed types: {', '.join(self.ALLOWED_EXTENSIONS)}")
            return result
        
        # File size validation
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > self.MAX_FILE_SIZE:
            result['errors'].append(f"File too large. Maximum size: {self.MAX_FILE_SIZE / 1024 / 1024:.1f}MB")
            return result
        
        if file_size == 0:
            result['errors'].append("Empty file")
            return result
        
        # MIME type validation
        mime_type = mimetypes.guess_type(filename)[0]
        allowed_mimes = {'text/csv', 'text/plain'}
        if mime_type not in allowed_mimes:
            result['errors'].append(f"Invalid file type. Expected CSV or text file")
            return result
        
        # File hash for integrity
        file_content = file.read()
        file.seek(0)  # Reset for later use
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        result['file_info'] = {
            'filename': filename,
            'size_bytes': file_size,
            'mime_type': mime_type,
            'hash': file_hash,
            'upload_time': datetime.utcnow().isoformat()
        }
        result['valid'] = True
        
        return result
    
    def _validate_cpet_content(self, file: FileStorage) -> Dict[str, Any]:
        """Scientific validation of CPET data content"""
        result = {'valid': False, 'errors': [], 'warnings': [], 'cpet_info': {}}
        
        try:
            # Read CSV content
            df = pd.read_csv(file, nrows=self.MAX_ROWS)
            file.seek(0)  # Reset file pointer
            
            if df.empty:
                result['errors'].append("File contains no data")
                return result
            
            # Check for required CPET columns
            columns = set(df.columns.str.strip().str.upper())
            missing_required = self.REQUIRED_COLUMNS - columns
            
            if missing_required:
                result['errors'].append(f"Missing required CPET columns: {', '.join(missing_required)}")
                return result
            
            # Validate data quality
            data_issues = self._validate_cpet_data_quality(df, columns)
            result['warnings'].extend(data_issues['warnings'])
            
            if data_issues['critical_errors']:
                result['errors'].extend(data_issues['critical_errors'])
                return result
            
            # Record CPET file characteristics for scientific validation
            result['cpet_info'] = {
                'row_count': len(df),
                'column_count': len(df.columns),
                'available_columns': list(columns),
                'missing_optional_columns': list(self.OPTIONAL_COLUMNS - columns),
                'data_range_minutes': self._estimate_test_duration(df),
                'data_quality_score': data_issues['quality_score']
            }
            
            result['valid'] = True
            
        except pd.errors.EmptyDataError:
            result['errors'].append("File appears to be empty or corrupted")
        except pd.errors.ParserError as e:
            result['errors'].append(f"CSV parsing error: {str(e)}")
        except Exception as e:
            result['errors'].append(f"Data validation error: {str(e)}")
        
        return result
    
    def _validate_cpet_data_quality(self, df: pd.DataFrame, columns: set) -> Dict[str, Any]:
        """Validate CPET data quality for scientific accuracy"""
        result = {'warnings': [], 'critical_errors': [], 'quality_score': 0}
        
        quality_checks = 0
        passed_checks = 0
        
        # Check for negative values in physiological parameters (should be positive)
        for col in ['VO2', 'VCO2', 'VE']:
            if col in columns:
                quality_checks += 1
                negative_count = (df[col] < 0).sum()
                if negative_count > len(df) * 0.1:  # More than 10% negative values
                    result['critical_errors'].append(f"Too many negative values in {col} ({negative_count} rows)")
                elif negative_count > 0:
                    result['warnings'].append(f"Some negative values in {col} ({negative_count} rows)")
                else:
                    passed_checks += 1
        
        # Check for missing data
        quality_checks += 1
        missing_percentage = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_percentage > 0.5:  # More than 50% missing data
            result['critical_errors'].append(f"Excessive missing data: {missing_percentage*100:.1f}%")
        elif missing_percentage > 0.2:  # More than 20% missing data
            result['warnings'].append(f"High missing data: {missing_percentage*100:.1f}%")
        else:
            passed_checks += 1
        
        # Check data consistency (basic physiological ranges)
        if 'VO2' in columns:
            quality_checks += 1
            vo2_values = df['VO2'].dropna()
            if len(vo2_values) > 0:
                if vo2_values.max() > 10000 or vo2_values.min() < 0:  # Reasonable VO2 range
                    result['warnings'].append("VO2 values outside typical physiological range")
                else:
                    passed_checks += 1
        
        result['quality_score'] = passed_checks / quality_checks if quality_checks > 0 else 0
        return result
    
    def _estimate_test_duration(self, df: pd.DataFrame) -> Optional[float]:
        """Estimate CPET test duration in minutes"""
        if 'Time' in df.columns:
            time_col = df['Time'].dropna()
            if len(time_col) > 0:
                return (time_col.max() - time_col.min()) / 60.0  # Convert to minutes
        return None
    
    def _allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed"""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.ALLOWED_EXTENSIONS


class MedicalDataManager:
    """Manager for medical data privacy compliance and automatic cleanup"""
    
    def __init__(self, upload_folder: str, cleanup_minutes: int, logger):
        self.upload_folder = upload_folder
        self.cleanup_minutes = cleanup_minutes
        self.logger = logger
        self.file_registry: Dict[str, Dict] = {}
        self.cleanup_thread = None
        self.running = False
        
        # Ensure upload folder exists
        os.makedirs(upload_folder, exist_ok=True)
        
        # Start cleanup thread
        self.start_cleanup_service()
    
    def register_file(self, file_path: str, session_id: str, metadata: Dict) -> None:
        """Register a file for automatic cleanup"""
        cleanup_time = datetime.utcnow() + timedelta(minutes=self.cleanup_minutes)
        
        self.file_registry[file_path] = {
            'session_id': session_id,
            'upload_time': datetime.utcnow(),
            'cleanup_time': cleanup_time,
            'metadata': metadata
        }
        
        self.logger.logger.debug(f"Registered file for cleanup: {file_path}")
    
    def save_temp_file(self, file: FileStorage, session_id: str) -> Tuple[str, str]:
        """Save uploaded file temporarily with security measures"""
        # Create secure temporary file
        filename = secure_filename(file.filename)
        file_path = os.path.join(self.upload_folder, f"{session_id}_{filename}")
        
        # Save file
        file.save(file_path)
        
        # Register for cleanup
        self.register_file(file_path, session_id, {
            'original_filename': filename,
            'size_bytes': os.path.getsize(file_path)
        })
        
        return file_path, filename
    
    def cleanup_expired_files(self) -> int:
        """Clean up expired files (medical data privacy compliance)"""
        current_time = datetime.utcnow()
        cleaned_count = 0
        files_to_remove = []
        
        for file_path, info in self.file_registry.items():
            if current_time >= info['cleanup_time']:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        cleaned_count += 1
                    files_to_remove.append(file_path)
                except Exception as e:
                    self.logger.log_security_event('file_cleanup_error', {
                        'file_path': file_path,
                        'error': str(e)
                    })
        
        # Remove from registry
        for file_path in files_to_remove:
            del self.file_registry[file_path]
        
        if cleaned_count > 0:
            self.logger.log_file_cleanup('system_cleanup', cleaned_count)
        
        return cleaned_count
    
    def cleanup_session_files(self, session_id: str) -> int:
        """Immediately clean up all files for a session"""
        cleaned_count = 0
        files_to_remove = []
        
        for file_path, info in self.file_registry.items():
            if info['session_id'] == session_id:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        cleaned_count += 1
                    files_to_remove.append(file_path)
                except Exception as e:
                    self.logger.log_security_event('session_cleanup_error', {
                        'session_id': session_id,
                        'file_path': file_path,
                        'error': str(e)
                    })
        
        # Remove from registry
        for file_path in files_to_remove:
            del self.file_registry[file_path]
        
        if cleaned_count > 0:
            self.logger.log_file_cleanup(session_id, cleaned_count)
        
        return cleaned_count
    
    def start_cleanup_service(self) -> None:
        """Start background cleanup service"""
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            return
        
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
    
    def stop_cleanup_service(self) -> None:
        """Stop background cleanup service"""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join()
    
    def _cleanup_worker(self) -> None:
        """Background worker for file cleanup"""
        while self.running:
            try:
                self.cleanup_expired_files()
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.log_security_event('cleanup_worker_error', {'error': str(e)})
                time.sleep(60)


def create_security_manager(config, logger):
    """Factory function to create security manager"""
    file_validator = CPETFileValidator(logger)
    
    # Handle different config types gracefully
    upload_folder = getattr(config, 'UPLOAD_FOLDER', 'temp_uploads')
    cleanup_minutes = getattr(config, 'AUTO_CLEANUP_MINUTES', 10)
    
    data_manager = MedicalDataManager(
        upload_folder,
        cleanup_minutes,
        logger
    )
    
    return {
        'file_validator': file_validator,
        'data_manager': data_manager
    }