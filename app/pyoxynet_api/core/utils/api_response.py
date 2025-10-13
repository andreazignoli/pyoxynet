"""
API Response Utilities
Standardized response format for PyOxynet API
"""
import json
import time
from typing import Any, Dict, List, Optional
from flask import jsonify, make_response
import pandas as pd
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy arrays and pandas objects"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        if isinstance(obj, pd.Series):
            return obj.to_list()
        return super().default(obj)


class APIResponse:
    """Standardized API response builder for PyOxynet"""
    
    API_VERSION = "1.0.0"
    
    @staticmethod
    def success(data: Any = None, 
                message: str = None, 
                metadata: Optional[Dict] = None,
                processing_time: Optional[float] = None,
                status_code: int = 200) -> tuple:
        """
        Create a standardized success response
        
        Args:
            data: Response data payload
            message: Optional success message
            metadata: Additional metadata
            processing_time: Processing time in seconds
            status_code: HTTP status code
            
        Returns:
            Tuple of (response, status_code)
        """
        response_data = {
            "success": True,
            "status": "ok",
            "data": data,
            "metadata": {
                "api_version": APIResponse.API_VERSION,
                "timestamp": pd.Timestamp.now().isoformat(),
                "processing_time_ms": round(processing_time * 1000, 2) if processing_time else None,
                **(metadata or {})
            },
            "errors": None
        }
        
        if message:
            response_data["message"] = message
            
        return response_data, status_code
    
    @staticmethod
    def error(message: str,
              errors: Optional[List[str]] = None,
              error_code: Optional[str] = None,
              details: Optional[Dict] = None,
              status_code: int = 400) -> tuple:
        """
        Create a standardized error response
        
        Args:
            message: Error message
            errors: List of detailed errors
            error_code: Machine-readable error code
            details: Additional error details
            status_code: HTTP status code
            
        Returns:
            Tuple of (response, status_code)
        """
        response_data = {
            "success": False,
            "status": "error",
            "data": None,
            "metadata": {
                "api_version": APIResponse.API_VERSION,
                "timestamp": pd.Timestamp.now().isoformat()
            },
            "error": {
                "message": message,
                "code": error_code,
                "details": details,
                "errors": errors or []
            }
        }
        
        return response_data, status_code
    
    @staticmethod
    def validation_error(validation_errors: Dict,
                        message: str = "Validation failed") -> tuple:
        """
        Create a standardized validation error response
        
        Args:
            validation_errors: Dictionary of field validation errors
            message: Main error message
            
        Returns:
            Tuple of (response, status_code)
        """
        return APIResponse.error(
            message=message,
            error_code="VALIDATION_ERROR",
            details={"field_errors": validation_errors},
            status_code=422
        )
    
    @staticmethod
    def not_found(resource: str = "Resource") -> tuple:
        """
        Create a standardized 404 response
        
        Args:
            resource: Name of the resource that wasn't found
            
        Returns:
            Tuple of (response, status_code)
        """
        return APIResponse.error(
            message=f"{resource} not found",
            error_code="NOT_FOUND",
            status_code=404
        )
    
    @staticmethod
    def unauthorized(message: str = "Authentication required") -> tuple:
        """
        Create a standardized 401 response
        
        Returns:
            Tuple of (response, status_code)
        """
        return APIResponse.error(
            message=message,
            error_code="UNAUTHORIZED",
            status_code=401
        )
    
    @staticmethod
    def forbidden(message: str = "Access forbidden") -> tuple:
        """
        Create a standardized 403 response
        
        Returns:
            Tuple of (response, status_code)
        """
        return APIResponse.error(
            message=message,
            error_code="FORBIDDEN",
            status_code=403
        )
    
    @staticmethod
    def rate_limited(retry_after: Optional[int] = None) -> tuple:
        """
        Create a standardized rate limit response
        
        Args:
            retry_after: Seconds until rate limit resets
            
        Returns:
            Tuple of (response, status_code)
        """
        response, status = APIResponse.error(
            message="Rate limit exceeded",
            error_code="RATE_LIMITED",
            details={"retry_after_seconds": retry_after} if retry_after else None,
            status_code=429
        )
        
        if retry_after:
            response = make_response(response)
            response.headers['Retry-After'] = str(retry_after)
            return response, status
            
        return response, status
    
    @staticmethod
    def internal_error(message: str = "Internal server error",
                      error_id: Optional[str] = None) -> tuple:
        """
        Create a standardized internal error response
        
        Args:
            message: Error message (generic for security)
            error_id: Internal error ID for tracking
            
        Returns:
            Tuple of (response, status_code)
        """
        return APIResponse.error(
            message=message,
            error_code="INTERNAL_ERROR",
            details={"error_id": error_id} if error_id else None,
            status_code=500
        )


class APIResponseTimer:
    """Context manager for measuring API response times"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0


class PaginatedResponse:
    """Helper for paginated API responses"""
    
    @staticmethod
    def create(items: List[Any],
               page: int,
               per_page: int,
               total_items: int,
               endpoint: str = None) -> Dict:
        """
        Create paginated response data
        
        Args:
            items: List of items for current page
            page: Current page number (1-based)
            per_page: Items per page
            total_items: Total number of items
            endpoint: API endpoint for pagination links
            
        Returns:
            Dictionary with paginated data
        """
        total_pages = (total_items + per_page - 1) // per_page
        
        pagination_data = {
            "items": items,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total_items": total_items,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1,
                "next_page": page + 1 if page < total_pages else None,
                "prev_page": page - 1 if page > 1 else None
            }
        }
        
        # Add pagination links if endpoint provided
        if endpoint:
            base_url = f"/api/v1/{endpoint.lstrip('/')}"
            pagination_data["pagination"]["links"] = {
                "first": f"{base_url}?page=1&per_page={per_page}",
                "last": f"{base_url}?page={total_pages}&per_page={per_page}",
                "next": f"{base_url}?page={page + 1}&per_page={per_page}" if page < total_pages else None,
                "prev": f"{base_url}?page={page - 1}&per_page={per_page}" if page > 1 else None
            }
        
        return pagination_data


# Common HTTP status codes with descriptions
HTTP_STATUS_CODES = {
    200: "OK",
    201: "Created", 
    202: "Accepted",
    204: "No Content",
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    406: "Not Acceptable",
    409: "Conflict",
    410: "Gone",
    413: "Request Entity Too Large",
    415: "Unsupported Media Type",
    422: "Unprocessable Entity",
    429: "Too Many Requests",
    500: "Internal Server Error",
    501: "Not Implemented",
    502: "Bad Gateway",
    503: "Service Unavailable"
}


# Common API error codes
API_ERROR_CODES = {
    # General errors
    "INTERNAL_ERROR": "An unexpected internal error occurred",
    "VALIDATION_ERROR": "Request validation failed",
    "NOT_FOUND": "Requested resource not found",
    "UNAUTHORIZED": "Authentication required",
    "FORBIDDEN": "Access forbidden",
    "RATE_LIMITED": "Rate limit exceeded",
    
    # File processing errors
    "NO_FILE": "No file provided in request",
    "EMPTY_FILE": "Uploaded file is empty",
    "FILE_TOO_LARGE": "File size exceeds maximum allowed",
    "UNSUPPORTED_FILE_TYPE": "File type not supported",
    "FILE_PROCESSING_ERROR": "Failed to process uploaded file",
    
    # CPET data errors
    "INVALID_CPET_DATA": "CPET data validation failed",
    "MISSING_REQUIRED_COLUMNS": "Required CPET columns missing",
    "INSUFFICIENT_DATA": "Insufficient data points for analysis",
    "DATA_QUALITY_ERROR": "Data quality issues detected",
    
    # Analysis errors
    "ANALYSIS_ERROR": "CPET analysis failed",
    "MODEL_ERROR": "ML model inference failed",
    "VISUALIZATION_ERROR": "Chart generation failed",
    
    # Service errors
    "SERVICE_UNAVAILABLE": "Service temporarily unavailable",
    "PROCESSING_TIMEOUT": "Analysis processing timeout",
    "QUOTA_EXCEEDED": "Usage quota exceeded"
}