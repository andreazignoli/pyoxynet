# PyOxynet API Testing Examples

These examples demonstrate the **exact** working formats for the PyOxynet API endpoints. All examples have been tested and verified to work.

## Prerequisites

1. **Start the unified API server:**
   ```bash
   python pyoxynet_api.py
   ```
   
2. **API will be available at:**
   - API info: `http://127.0.0.1:5000/info` (comprehensive information and examples)
   - Main endpoint: `http://127.0.0.1:5000/analyze/data`
   - Health check: `http://127.0.0.1:5000/health`
   - Documentation: `http://127.0.0.1:5000/docs/`

## Test Commands

### Example 1: Minimal Format (Direct Data Array)

This is the simplest format using only required fields:

```bash
curl -X POST http://127.0.0.1:5000/analyze/data \
  -H "Content-Type: application/json" \
  -d @example_request_minimal.json
```

**File: `example_request_minimal.json`**
- Contains 50 data points (minimum 40 required)
- Uses only required fields: VO2, VCO2, VE, PetO2, PetCO2
- Results in 11 predictions with moving window analysis coverage: "11/50 points"

### Example 2: Complete Dataset Format

This format includes optional metadata and time information:

```bash
curl -X POST http://127.0.0.1:5000/analyze/data \
  -H "Content-Type: application/json" \
  -d @example_request_dataset.json
```

**File: `example_request_dataset.json`**
- Contains complete dataset with metadata (id, VO2max)
- Includes time stamps (t field)
- Same data as Example 1 but in dataset wrapper format

### Example 3: Test with Large Dataset

Using the full test data (361 points):

```bash
curl -X POST http://127.0.0.1:5000/analyze/data \
  -H "Content-Type: application/json" \
  -d '{"datasets": '$(cat test_data/test_data_ET.json)'}'
```

Results in 322 predictions with coverage: "322/361 points"

## Expected Response Format

Both examples return the same structure:

```json
{
  "success": true,
  "status": "ok", 
  "data": {
    "success": true,
    "timestamp": "2025-09-26T15:56:00.402962",
    "ml_analysis": {
      "domain_probabilities": {
        "Moderate": 0.175,
        "Heavy": 0.348, 
        "Severe": 0.477
      },
      "dominant_domain": "Severe",
      "confidence": 0.477,
      "ventilatory_thresholds": {
        "VT1": null,
        "VT2": null  
      },
      "threshold_detection": {
        "VT1_detected": false,
        "VT2_detected": false
      },
      "temporal_analysis": {
        "predictions_over_time": [...],
        "time_points": 11,
        "dominant_domains_over_time": [2,2,2,...],
        "time_indices": [39,40,41,...]
      },
      "moving_window_info": {
        "window_size": 40,
        "total_windows": 11,
        "analysis_coverage": "11/50 points"
      }
    },
    "visualizations": {...},
    "analysis_report": {...}
  }
}
```

## Key Features Demonstrated

1. **Moving Window Analysis**: Uses 40-point sliding window for threshold detection
2. **Domain Classification**: Classifies as Moderate/Heavy/Severe domains
3. **Temporal Analysis**: Shows how domain probabilities change over time
4. **Quality Assessment**: Provides data quality metrics and recommendations
5. **Visualizations**: Generates interactive plots for VO2 trends and domain transitions

## Error Testing

### Insufficient Data
```bash
curl -X POST http://127.0.0.1:5000/analyze/data \
  -H "Content-Type: application/json" \
  -d '{"data": [{"VO2": 1000, "VCO2": 800, "VE": 25, "PetO2": 75, "PetCO2": 35}]}'
```

Expected error: "Minimum 40 data points required"

### Missing Required Fields
```bash
curl -X POST http://127.0.0.1:5000/analyze/data \
  -H "Content-Type: application/json" \
  -d '{"data": [{"VO2": 1000}]}'
```

Expected error: Validation error for missing required fields.

## Units Reference

- **VO2, VCO2**: ml/min (milliliters per minute)
- **VE**: L/min (liters per minute)  
- **PetO2, PetCO2**: mmHg (millimeters of mercury)
- **t**: seconds (optional)

All examples use these exact units and have been verified to execute successfully.