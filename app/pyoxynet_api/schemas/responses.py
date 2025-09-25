"""
Response schemas for PyOxynet API
Pydantic models for response validation and documentation
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from datetime import datetime


class VentilatorThreshold(BaseModel):
    """Ventilatory threshold information"""
    vo2_absolute: Optional[float] = Field(None, description="VO2 at threshold (ml/min)")
    vo2_percent_max: Optional[float] = Field(None, description="Threshold as % of VO2max")
    detected: bool = Field(..., description="Whether threshold was detected")


class DomainProbabilities(BaseModel):
    """Exercise domain probabilities"""
    Moderate: float = Field(..., description="Probability of moderate domain", ge=0, le=1)
    Heavy: float = Field(..., description="Probability of heavy domain", ge=0, le=1)
    Severe: float = Field(..., description="Probability of severe domain", ge=0, le=1)


class MLAnalysisResult(BaseModel):
    """ML analysis results"""
    domain_probabilities: DomainProbabilities = Field(..., description="Domain classification probabilities")
    dominant_domain: str = Field(..., description="Most likely exercise domain")
    confidence: float = Field(..., description="Overall confidence in prediction", ge=0, le=1)
    ventilatory_thresholds: Dict[str, Optional[float]] = Field(..., description="VT1 and VT2 estimates")
    data_quality: Dict[str, Any] = Field(..., description="Data quality assessment")


class AnalysisSummary(BaseModel):
    """Analysis summary information"""
    vo2_max: Optional[float] = Field(None, description="Maximum VO2 (ml/min)")
    vo2_max_units: str = Field("ml/min", description="Units for VO2 measurements")
    dominant_domain: str = Field(..., description="Primary exercise domain")
    confidence: float = Field(..., description="Analysis confidence", ge=0, le=1)
    test_duration_minutes: Optional[float] = Field(None, description="Total test duration")
    data_quality: str = Field(..., description="Overall data quality rating")


class ThresholdAnalysis(BaseModel):
    """Ventilatory threshold analysis"""
    vt1: VentilatorThreshold = Field(..., description="First ventilatory threshold")
    vt2: VentilatorThreshold = Field(..., description="Second ventilatory threshold")


class Interpretation(BaseModel):
    """Clinical interpretation of results"""
    primary_finding: str = Field(..., description="Main analysis finding")
    threshold_assessment: str = Field(..., description="Threshold detection assessment")
    clinical_significance: str = Field(..., description="Clinical significance")
    recommendations: List[str] = Field(..., description="Recommendations based on results")


class DataQualityAssessment(BaseModel):
    """Data quality assessment"""
    overall_quality: str = Field(..., description="Overall quality rating")
    total_data_points: int = Field(..., description="Total number of data points")
    missing_data_percentage: float = Field(..., description="Percentage of missing data")
    validation_warnings: List[str] = Field(..., description="Data validation warnings")
    recommendations: List[str] = Field(..., description="Data quality recommendations")


class TechnicalDetails(BaseModel):
    """Technical analysis details"""
    metabolimeter_type: str = Field(..., description="Detected metabolimeter type")
    sampling_rate_hz: Optional[float] = Field(None, description="Data sampling rate")
    analysis_timestamp: str = Field(..., description="Analysis timestamp (ISO format)")
    ml_model_info: Dict[str, Any] = Field(..., description="ML model information")


class AnalysisReport(BaseModel):
    """Comprehensive analysis report"""
    summary: AnalysisSummary = Field(..., description="Analysis summary")
    thresholds: ThresholdAnalysis = Field(..., description="Threshold analysis")
    domain_analysis: DomainProbabilities = Field(..., description="Domain probabilities")
    interpretation: Interpretation = Field(..., description="Clinical interpretation")
    data_quality_assessment: DataQualityAssessment = Field(..., description="Data quality assessment")
    technical_details: TechnicalDetails = Field(..., description="Technical details")


class VisualizationResult(BaseModel):
    """Single visualization result"""
    success: bool = Field(..., description="Whether visualization was generated successfully")
    figure: Optional[Dict[str, Any]] = Field(None, description="Plotly figure JSON")
    error: Optional[str] = Field(None, description="Error message if failed")


class ProcessingInfo(BaseModel):
    """Processing information"""
    total_data_points: int = Field(..., description="Total data points processed")
    analysis_duration: Optional[Union[float, str]] = Field(None, description="Test duration")
    data_quality: str = Field(..., description="Data quality rating")


class ValidationResult(BaseModel):
    """Data validation result"""
    valid: bool = Field(..., description="Whether data is valid for analysis")
    errors: List[str] = Field(..., description="Validation errors")
    warnings: List[str] = Field(..., description="Validation warnings")
    recommendations: List[str] = Field(..., description="Recommendations")


class DataPreview(BaseModel):
    """Data preview information"""
    rows: List[Dict[str, Any]] = Field(..., description="Preview data rows")
    total_rows: int = Field(..., description="Total rows in dataset")
    columns: List[str] = Field(..., description="Available columns")


class DataMetadata(BaseModel):
    """Data metadata information"""
    filename: str = Field(..., description="Original filename")
    metabolimeter_type: str = Field(..., description="Detected metabolimeter type")
    total_rows: int = Field(..., description="Total rows")
    columns: List[str] = Field(..., description="Available columns")
    duration_minutes: Optional[float] = Field(None, description="Test duration in minutes")
    data_quality: str = Field(..., description="Data quality assessment")
    available_variables: Dict[str, bool] = Field(..., description="Available CPET variables")


class CPETAnalysisResponse(BaseModel):
    """Complete CPET analysis response"""
    success: bool = Field(..., description="Analysis success status")
    data: Optional[Dict[str, Any]] = Field(None, description="Analysis data")
    visualizations: Optional[Dict[str, VisualizationResult]] = Field(None, description="Generated visualizations")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Response metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "data": {
                    "session_id": "session_123",
                    "filename": "test.csv",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "ml_analysis": {
                        "domain_probabilities": {
                            "Moderate": 0.6,
                            "Heavy": 0.3,
                            "Severe": 0.1
                        },
                        "dominant_domain": "Moderate",
                        "confidence": 0.85
                    },
                    "analysis_report": {
                        "summary": {
                            "vo2_max": 2500,
                            "dominant_domain": "Moderate",
                            "confidence": 0.85,
                            "data_quality": "Good"
                        }
                    }
                },
                "visualizations": {
                    "vo2_time_plot": {
                        "success": True,
                        "figure": {}
                    }
                },
                "metadata": {
                    "api_version": "1.0.0",
                    "processing_time_ms": 1250
                }
            }
        }


class ValidationResponse(BaseModel):
    """File validation response"""
    success: bool = Field(..., description="Validation success status")
    validation: ValidationResult = Field(..., description="Validation results")
    metadata: DataMetadata = Field(..., description="Data metadata")
    preview: DataPreview = Field(..., description="Data preview")


class HealthCheckResponse(BaseModel):
    """Health check response"""
    healthy: bool = Field(..., description="Overall health status")
    status: str = Field(..., description="Health status description")
    services: Dict[str, Any] = Field(..., description="Individual service status")
    timestamp: str = Field(..., description="Check timestamp")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Error response format"""
    success: bool = Field(False, description="Always false for errors")
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    code: str = Field(..., description="Error code")
    errors: Optional[List[str]] = Field(None, description="Detailed errors")
    timestamp: Optional[str] = Field(None, description="Error timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "error": "Validation Error",
                "message": "Missing required CPET columns",
                "code": "MISSING_COLUMNS",
                "errors": ["Missing column: VO2", "Missing column: VCO2"],
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }