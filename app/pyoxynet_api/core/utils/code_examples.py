"""
Code Examples Generator for PyOxynet API
Generate code examples in multiple languages for documentation
"""
from typing import Dict, List, Any
import json


class CodeExampleGenerator:
    """Generate code examples for PyOxynet API in multiple languages"""
    
    BASE_URL = "https://api.pyoxynet.com"
    
    @staticmethod
    def curl_file_analysis(filename: str = "cpet_data.csv", 
                          options: Dict = None) -> str:
        """Generate cURL example for file analysis"""
        options_json = json.dumps(options or {"include_nine_panel": True})
        
        return f'''# Analyze CPET data file with cURL
curl -X POST "{CodeExampleGenerator.BASE_URL}/api/v1/analyze/file" \\
     -H "Content-Type: multipart/form-data" \\
     -F "file=@{filename}" \\
     -F 'options={options_json}' \\
     -F "session_id=my_analysis_session"'''
    
    @staticmethod
    def curl_json_analysis(sample_data: List[Dict] = None) -> str:
        """Generate cURL example for JSON data analysis"""
        if sample_data is None:
            sample_data = [
                {"TIME": 0, "VO2": 500, "VCO2": 400, "VE": 15, "HR": 60},
                {"TIME": 30, "VO2": 800, "VCO2": 720, "VE": 25, "HR": 80},
                {"TIME": 60, "VO2": 1200, "VCO2": 1080, "VE": 35, "HR": 100}
            ]
        
        request_data = {
            "data": sample_data,
            "options": {
                "include_nine_panel": True,
                "threshold_detection_method": "ml",
                "confidence_threshold": 0.7
            }
        }
        
        return f'''# Analyze CPET JSON data with cURL
curl -X POST "{CodeExampleGenerator.BASE_URL}/api/v1/analyze/data" \\
     -H "Content-Type: application/json" \\
     -d '{json.dumps(request_data, indent=2)}' '''
    
    @staticmethod
    def python_sdk_example() -> str:
        """Generate Python SDK example"""
        return '''# PyOxynet Python SDK Example
import pyoxynet
import pandas as pd

# Initialize client
client = pyoxynet.Client(
    api_key="your_api_key",  # Optional for public access
    base_url="https://api.pyoxynet.com"
)

# Method 1: Analyze file
try:
    # Upload and analyze CPET file
    results = client.analyze_file(
        filepath="cpet_data.csv",
        options={
            "include_nine_panel": True,
            "threshold_detection_method": "ml",
            "confidence_threshold": 0.8
        }
    )
    
    # Access results
    print(f"Analysis completed successfully!")
    print(f"Dominant domain: {results.ml_analysis.dominant_domain}")
    print(f"Confidence: {results.ml_analysis.confidence:.2%}")
    print(f"VO2 max: {results.analysis_report.summary.vo2_max} ml/min")
    
    # Access thresholds
    vt1 = results.ml_analysis.ventilatory_thresholds.get('VT1')
    vt2 = results.ml_analysis.ventilatory_thresholds.get('VT2')
    if vt1:
        print(f"VT1: {vt1} ml/min")
    if vt2:
        print(f"VT2: {vt2} ml/min")
    
    # Save visualizations
    if 'vo2_time_plot' in results.visualizations:
        results.save_plot('vo2_time_plot', 'vo2_analysis.png')
    
except pyoxynet.ValidationError as e:
    print(f"Data validation failed: {e.message}")
    for error in e.errors:
        print(f"  - {error}")
        
except pyoxynet.ProcessingError as e:
    print(f"Analysis failed: {e.message}")

# Method 2: Analyze DataFrame directly
data = pd.DataFrame({
    'TIME': [0, 30, 60, 90, 120],
    'VO2': [500, 800, 1200, 1600, 2000],
    'VCO2': [400, 720, 1080, 1440, 1800],
    'VE': [15, 25, 35, 45, 55],
    'HR': [60, 80, 100, 120, 140]
})

results = client.analyze_data(
    data=data,
    options={"include_metabolic_analysis": True}
)

# Export results
results.export_report("cpet_analysis_report.pdf", format="pdf")
results.export_data("cpet_results.json", format="json")

# Batch processing
files = ["subject1.csv", "subject2.csv", "subject3.csv"]
batch_results = client.analyze_batch(files)

for filename, result in batch_results.items():
    if result.success:
        print(f"{filename}: {result.ml_analysis.dominant_domain}")
    else:
        print(f"{filename}: Analysis failed - {result.error}")'''
    
    @staticmethod
    def r_integration_example() -> str:
        """Generate R integration example"""
        return '''# PyOxynet R Integration Example
library(httr)
library(jsonlite)
library(readr)

# Configuration
base_url <- "https://api.pyoxynet.com"
api_key <- "your_api_key"  # Optional

# Function to analyze CPET file
analyze_cpet_file <- function(filepath, options = list()) {
  url <- paste0(base_url, "/api/v1/analyze/file")
  
  # Prepare request
  body <- list(
    file = upload_file(filepath),
    options = toJSON(options, auto_unbox = TRUE)
  )
  
  # Make request
  response <- POST(
    url = url,
    body = body,
    encode = "multipart",
    add_headers(
      "X-API-Key" = api_key
    )
  )
  
  # Parse response
  if (status_code(response) == 200) {
    result <- fromJSON(content(response, "text"))
    return(result$data)
  } else {
    stop(paste("API Error:", content(response, "text")))
  }
}

# Function to analyze data frame
analyze_cpet_data <- function(data, options = list()) {
  url <- paste0(base_url, "/api/v1/analyze/data")
  
  # Prepare request body
  request_data <- list(
    data = data,
    options = options
  )
  
  # Make request
  response <- POST(
    url = url,
    body = toJSON(request_data, auto_unbox = TRUE),
    add_headers(
      "Content-Type" = "application/json",
      "X-API-Key" = api_key
    )
  )
  
  # Parse response
  if (status_code(response) == 200) {
    result <- fromJSON(content(response, "text"))
    return(result$data)
  } else {
    stop(paste("API Error:", content(response, "text")))
  }
}

# Example usage
tryCatch({
  # Analyze file
  results <- analyze_cpet_file(
    "cpet_data.csv",
    options = list(
      include_nine_panel = TRUE,
      threshold_detection_method = "ml",
      confidence_threshold = 0.8
    )
  )
  
  # Extract results
  ml_results <- results$analysis$ml_results
  report <- results$analysis$report
  
  cat("Analysis Results:\\n")
  cat("Dominant Domain:", ml_results$dominant_domain, "\\n")
  cat("Confidence:", sprintf("%.1f%%", ml_results$confidence * 100), "\\n")
  cat("VO2 Max:", report$summary$vo2_max, "ml/min\\n")
  
  # Plot results (if plotting library available)
  if (requireNamespace("plotly", quietly = TRUE)) {
    # Convert visualization to plotly
    vo2_plot_data <- results$visualizations$vo2_time_plot$figure
    p <- plotly::plot_ly() %>%
      plotly::add_trace(
        data = vo2_plot_data,
        type = "scatter",
        mode = "lines"
      )
    print(p)
  }
  
}, error = function(e) {
  cat("Error:", conditionMessage(e), "\\n")
})

# Analyze data frame
cpet_data <- data.frame(
  TIME = seq(0, 600, 30),
  VO2 = seq(500, 2500, 100),
  VCO2 = seq(400, 2200, 90),
  VE = seq(15, 75, 3),
  HR = seq(60, 180, 6)
)

results <- analyze_cpet_data(
  cpet_data,
  options = list(include_metabolic_analysis = TRUE)
)

# Save results
write_json(results, "cpet_analysis.json", pretty = TRUE)

# Batch processing function
analyze_cpet_batch <- function(file_list) {
  results <- list()
  
  for (file in file_list) {
    cat("Processing:", file, "\\n")
    tryCatch({
      result <- analyze_cpet_file(file)
      results[[file]] <- result
    }, error = function(e) {
      cat("Failed to process", file, ":", conditionMessage(e), "\\n")
      results[[file]] <- list(error = conditionMessage(e))
    })
  }
  
  return(results)
}

# Process multiple files
files <- c("subject1.csv", "subject2.csv", "subject3.csv")
batch_results <- analyze_cpet_batch(files)'''
    
    @staticmethod
    def javascript_example() -> str:
        """Generate JavaScript/Node.js example"""
        return '''// PyOxynet JavaScript/Node.js Example
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

class PyOxynetClient {
  constructor(options = {}) {
    this.baseURL = options.baseURL || 'https://api.pyoxynet.com';
    this.apiKey = options.apiKey;
    this.timeout = options.timeout || 30000;
  }

  // Analyze CPET file
  async analyzeFile(filePath, options = {}) {
    const form = new FormData();
    form.append('file', fs.createReadStream(filePath));
    form.append('options', JSON.stringify(options));
    
    if (options.sessionId) {
      form.append('session_id', options.sessionId);
    }

    try {
      const response = await axios.post(
        `${this.baseURL}/api/v1/analyze/file`,
        form,
        {
          headers: {
            ...form.getHeaders(),
            ...(this.apiKey && { 'X-API-Key': this.apiKey })
          },
          timeout: this.timeout
        }
      );

      return this.handleResponse(response);
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // Analyze JSON data
  async analyzeData(data, options = {}) {
    const requestData = {
      data: data,
      options: options
    };

    try {
      const response = await axios.post(
        `${this.baseURL}/api/v1/analyze/data`,
        requestData,
        {
          headers: {
            'Content-Type': 'application/json',
            ...(this.apiKey && { 'X-API-Key': this.apiKey })
          },
          timeout: this.timeout
        }
      );

      return this.handleResponse(response);
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // Validate file
  async validateFile(filePath, previewRows = 10) {
    const form = new FormData();
    form.append('file', fs.createReadStream(filePath));
    form.append('preview_rows', previewRows.toString());

    try {
      const response = await axios.post(
        `${this.baseURL}/api/v1/validate`,
        form,
        {
          headers: {
            ...form.getHeaders(),
            ...(this.apiKey && { 'X-API-Key': this.apiKey })
          }
        }
      );

      return this.handleResponse(response);
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // Health check
  async healthCheck() {
    try {
      const response = await axios.get(`${this.baseURL}/api/v1/health`);
      return this.handleResponse(response);
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // Response handler
  handleResponse(response) {
    if (response.data.success) {
      return {
        success: true,
        data: response.data.data,
        metadata: response.data.metadata,
        message: response.data.message
      };
    } else {
      throw new Error(response.data.error?.message || 'API request failed');
    }
  }

  // Error handler
  handleError(error) {
    if (error.response) {
      const errorData = error.response.data;
      const message = errorData.error?.message || 'API request failed';
      const code = errorData.error?.code || 'UNKNOWN_ERROR';
      
      const apiError = new Error(message);
      apiError.code = code;
      apiError.status = error.response.status;
      apiError.details = errorData.error?.details;
      
      return apiError;
    }
    
    return error;
  }
}

// Usage examples
async function main() {
  const client = new PyOxynetClient({
    apiKey: 'your_api_key',  // Optional
    timeout: 60000  // 60 seconds
  });

  try {
    // Example 1: Analyze file
    console.log('Analyzing CPET file...');
    const fileResults = await client.analyzeFile('cpet_data.csv', {
      include_nine_panel: true,
      threshold_detection_method: 'ml',
      confidence_threshold: 0.8,
      sessionId: 'my_session_' + Date.now()
    });

    console.log('File Analysis Results:');
    console.log('Dominant Domain:', fileResults.data.analysis.ml_results.dominant_domain);
    console.log('Confidence:', (fileResults.data.analysis.ml_results.confidence * 100).toFixed(1) + '%');
    
    // Example 2: Analyze JSON data
    const sampleData = [
      { TIME: 0, VO2: 500, VCO2: 400, VE: 15, HR: 60 },
      { TIME: 30, VO2: 800, VCO2: 720, VE: 25, HR: 80 },
      { TIME: 60, VO2: 1200, VCO2: 1080, VE: 35, HR: 100 },
      { TIME: 90, VO2: 1600, VCO2: 1440, VE: 45, HR: 120 }
    ];

    console.log('\\nAnalyzing JSON data...');
    const dataResults = await client.analyzeData(sampleData, {
      include_metabolic_analysis: true
    });

    console.log('JSON Analysis Results:');
    console.log('VO2 Max:', dataResults.data.analysis.report.summary.vo2_max, 'ml/min');
    
    // Example 3: Validate file
    console.log('\\nValidating file...');
    const validation = await client.validateFile('cpet_data.csv', 5);
    
    console.log('Validation Results:');
    console.log('Valid:', validation.data.validation.valid);
    console.log('Data Quality:', validation.data.metadata.data_quality);
    
    // Example 4: Health check
    const health = await client.healthCheck();
    console.log('\\nHealth Status:', health.data.status);

  } catch (error) {
    console.error('Error:', error.message);
    if (error.code) {
      console.error('Error Code:', error.code);
    }
    if (error.details) {
      console.error('Details:', error.details);
    }
  }
}

// Run examples
if (require.main === module) {
  main().catch(console.error);
}

module.exports = PyOxynetClient;'''
    
    @staticmethod
    def get_all_examples() -> Dict[str, str]:
        """Get all code examples"""
        return {
            "curl_file": CodeExampleGenerator.curl_file_analysis(),
            "curl_json": CodeExampleGenerator.curl_json_analysis(),
            "python": CodeExampleGenerator.python_sdk_example(),
            "r": CodeExampleGenerator.r_integration_example(),
            "javascript": CodeExampleGenerator.javascript_example()
        }
    
    @staticmethod
    def generate_readme_examples() -> str:
        """Generate examples section for README"""
        return '''## Code Examples

### cURL
```bash
''' + CodeExampleGenerator.curl_file_analysis() + '''
```

### Python
```python
''' + CodeExampleGenerator.python_sdk_example()[:1000] + '''
# ... (see full example in documentation)
```

### R
```r
''' + CodeExampleGenerator.r_integration_example()[:800] + '''
# ... (see full example in documentation)
```

### JavaScript/Node.js
```javascript
''' + CodeExampleGenerator.javascript_example()[:600] + '''
// ... (see full example in documentation)
```

For complete examples and SDKs, visit: https://docs.pyoxynet.com'''