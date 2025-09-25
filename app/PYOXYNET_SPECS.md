# PyOxynet Modernization Specifications
## Critical Requirements Shared Across All Implementation Phases

### 🐍 Python Environment Specifications
- **Python Version**: 3.10.x (Required for TensorFlow Lite compatibility)
- **Virtual Environment**: Always use isolated environments
- **Package Management**: pip with --extra-index-url https://google-coral.github.io/py-repo/

### 🧠 TensorFlow Lite Requirements (NON-NEGOTIABLE)
- **Package**: `pyoxynet[tflite]==0.1.8` 
- **Runtime**: `tflite-runtime` from Google Coral repository
- **NumPy Compatibility**: `numpy==1.26.4` (pinned for TFLite compatibility)
- **Extra Index URL**: `--extra-index-url https://google-coral.github.io/py-repo/`

### 🔬 Scientific Data Requirements
- **CPET Data Format**: CSV with minimum columns: VO2, VCO2, VE
- **Optional Columns**: HR, RF, PetO2, PetCO2, VEVO2, VEVCO2, Time
- **Units**: Standard physiological units (ml/min/kg for VO2/VCO2, L/min for VE)
- **Data Quality**: Validate ranges, handle missing data, detect anomalies

### 🏥 Medical Data Privacy (NON-NEGOTIABLE)
- **Storage Policy**: NO persistent storage of medical data
- **Cleanup Policy**: Automatic cleanup within 60 minutes (production) / 10 minutes (development)
- **Processing**: All CPET data processed temporarily only
- **Logging**: Medical data sanitized from all logs

### 🔒 Security Specifications
- **Secret Management**: Environment variables only, never hardcoded
- **File Validation**: Type, size, content validation for all uploads
- **Input Sanitization**: All user inputs validated and sanitized
- **Error Handling**: No sensitive information in error responses

### 🌐 API Design Standards
- **Versioning**: `/api/v1/` prefix for all endpoints
- **Response Format**: Consistent JSON responses with success/error structure
- **Documentation**: OpenAPI/Swagger documentation required
- **Authentication**: JWT + API key authentication (Phase 4)

### 📊 Performance Requirements
- **Batch Processing**: Up to 100 CPET files per batch
- **Memory Limit**: 2GB maximum per batch processing job
- **Processing Time**: 100 files within 5 minutes target
- **File Size**: 10MB maximum per individual CPET file

### 🏗️ Architecture Principles
- **Separation of Concerns**: Config, Security, Logging, Business Logic separated
- **TDD Approach**: Tests before implementation (mandatory)
- **Error Recovery**: Graceful degradation and cleanup on failures
- **Monitoring**: Structured logging with scientific context

### 🚀 Deployment Specifications
- **Target Platform**: Heroku (primary), Docker compatible
- **WSGI Server**: Gunicorn for production
- **Environment Variables**: 
  - `SECRET_KEY` (required in production)
  - `FLASK_ENV` (development/production/testing)
  - `LOG_LEVEL` (DEBUG/INFO/WARNING/ERROR)

### 📦 Dependency Management
```bash
# Production Installation
pip install -r requirements.txt --extra-index-url https://google-coral.github.io/py-repo/

# Development Installation  
pip install -r requirements-dev.txt --extra-index-url https://google-coral.github.io/py-repo/
```

### 🧪 Testing Standards
- **Framework**: pytest with coverage reporting
- **Coverage Target**: 80% minimum
- **Test Categories**: Unit, Integration, Security, Scientific validation
- **Test Data**: Synthetic CPET data only (never real patient data)

### 🔄 Phase Implementation Order
1. **Phase 1**: Foundation & Security (Current)
2. **Phase 2**: Architectural Refactoring  
3. **Phase 3**: Modern API Design
4. **Phase 4**: Authentication & Rate Limiting
5. **Phase 5**: Modern Frontend Experience
6. **Phase 6**: Production Deployment & DevOps
7. **Phase 7**: Performance & Scalability
8. **Phase 8**: Business Readiness

---

## ⚠️ Critical Reminders for All Development Work

1. **Always use the Google Coral extra index** for TensorFlow Lite dependencies
2. **Never store or log medical data** - privacy is non-negotiable
3. **Python 3.10 compatibility** - newer versions may break TFLite
4. **Test with real CPET data structure** but synthetic values only
5. **Scientific accuracy validation** must be preserved through all changes

---

*This specification document must be referenced and followed in all PyOxynet modernization work.*