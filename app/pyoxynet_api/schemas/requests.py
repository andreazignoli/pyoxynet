"""
Request schemas for PyOxynet API
Pydantic models for request validation
"""
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
import pandas as pd


class CPETDataPoint(BaseModel):
    """Single CPET data point"""
    TIME: Optional[float] = Field(None, description="Time in seconds")
    VO2: float = Field(..., description="Oxygen uptake in ml/min", gt=0)
    VCO2: float = Field(..., description="Carbon dioxide output in ml/min", gt=0)
    VE: float = Field(..., description="Minute ventilation in L/min", gt=0)
    HR: Optional[float] = Field(None, description="Heart rate in bpm", gt=0, le=300)
    RER: Optional[float] = Field(None, description="Respiratory exchange ratio", gt=0.5, lt=2.0)
    LOAD: Optional[float] = Field(None, description="Workload in watts", ge=0)
    
    @validator('RER', pre=True, always=True)
    def calculate_rer_if_missing(cls, v, values):
        """Calculate RER from VO2 and VCO2 if not provided"""
        if v is None and 'VO2' in values and 'VCO2' in values:
            return values['VCO2'] / values['VO2']
        return v


class AnalysisOptions(BaseModel):
    """Analysis configuration options"""
    include_nine_panel: bool = Field(True, description="Include comprehensive 9-panel plot")
    include_metabolic_analysis: bool = Field(True, description="Include metabolic analysis plots")
    threshold_detection_method: str = Field('ml', description="Method for threshold detection")
    domain_classification: bool = Field(True, description="Perform exercise domain classification")
    generate_pdf_report: bool = Field(False, description="Generate PDF report")
    confidence_threshold: float = Field(0.7, description="Minimum confidence for predictions", ge=0.0, le=1.0)
    
    @validator('threshold_detection_method')
    def validate_threshold_method(cls, v):
        allowed_methods = ['ml', 'traditional', 'hybrid']
        if v not in allowed_methods:
            raise ValueError(f'Method must be one of: {allowed_methods}')
        return v


class CPETAnalysisRequest(BaseModel):
    """Request model for CPET data analysis"""
    data: List[CPETDataPoint] = Field(..., description="Array of CPET data points")
    options: Optional[AnalysisOptions] = Field(AnalysisOptions(), description="Analysis options")
    metadata: Optional[Dict[str, Any]] = Field({}, description="Additional metadata")
    
    @validator('data')
    def validate_data_length(cls, v):
        if len(v) < 10:
            raise ValueError('Minimum 10 data points required for analysis')
        if len(v) > 10000:
            raise ValueError('Maximum 10,000 data points allowed')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "data": [
                    {
                        "TIME": 0,
                        "VO2": 500,
                        "VCO2": 400,
                        "VE": 15,
                        "HR": 60
                    },
                    {
                        "TIME": 30,
                        "VO2": 800,
                        "VCO2": 720,
                        "VE": 25,
                        "HR": 80
                    }
                ],
                "options": {
                    "include_nine_panel": True,
                    "threshold_detection_method": "ml",
                    "confidence_threshold": 0.7
                },
                "metadata": {
                    "subject_id": "SUBJ001",
                    "test_protocol": "ramp"
                }
            }
        }


class FileUploadRequest(BaseModel):
    """Request model for file upload analysis"""
    session_id: Optional[str] = Field(None, description="Session identifier")
    options: Optional[AnalysisOptions] = Field(AnalysisOptions(), description="Analysis options")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "session_123",
                "options": {
                    "include_nine_panel": True,
                    "generate_pdf_report": False
                }
            }
        }


class ValidationRequest(BaseModel):
    """Request model for data validation only"""
    preview_rows: int = Field(10, description="Number of rows to include in preview", ge=1, le=100)
    
    class Config:
        schema_extra = {
            "example": {
                "preview_rows": 10
            }
        }