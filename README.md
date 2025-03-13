# Whole Cost Dewatering Analysis Tool

A comprehensive Python-based tool for analyzing wastewater dewatering processes and calculating associated costs based on parameters from the Whole Cost Dewatering Phase I Assessment.

## Overview

The Whole Cost Dewatering Analysis Tool helps calculate and analyze:
- Wet tons based on dewatered cake total solids percentage (TS%)
- Transportation costs based on fuel, distance, and tonnage
- Labor expenses with step functions for adding operators and mechanics
- Equipment expenses for additional tractors and trailers
- Polymer usage and costs for both dry and emulsion polymers
- Total costs to determine optimal operating points
- Sensitivity analysis across different TS% values (18%, 20%, 22%, 24%)

## Requirements

- Docker (recommended for easy setup)
- Python 3.9+ (if running locally)
- Dependencies: pandas, numpy, matplotlib, openpyxl

## Running with Docker (Recommended)

### 1. Build the Docker Image

```bash
docker build -t dewatering-tool .
```

### 2. Run the Tool

#### Using the Shell Script (Easiest)

We provide a convenient shell script to run the tool with various options:

```bash
# Make the script executable (first time only)
chmod +x run_dewatering_tool.sh

# Run with default settings
./run_dewatering_tool.sh

# Run with custom parameters and export results
./run_dewatering_tool.sh --dry-tons 25000 --cake-ts-target 22.0 --export-excel --sensitivity
```

Run `./run_dewatering_tool.sh --help` to see all available options.

#### Using Docker Directly

Basic usage:
```bash
docker run --rm -v $(pwd)/output:/app/output dewatering-tool
```

This will run the analysis with default parameters and output the results to your console.

### 3. Export Results to Excel

To export results to an Excel file in the output directory:
```bash
docker run --rm -v $(pwd)/output:/app/output dewatering-tool --export-excel
```

The results will be saved to `/app/output/dewatering_analysis_results.xlsx`.

### 4. Running Sensitivity Analysis

To run the sensitivity analysis with default TS% values (18%, 20%, 22%, 24%):
```bash
docker run --rm -v $(pwd)/output:/app/output dewatering-tool --sensitivity
```

To specify custom TS% values for sensitivity analysis:
```bash
docker run --rm -v $(pwd)/output:/app/output dewatering-tool --sensitivity --sensitivity-values 19,20,21,22
```

The sensitivity analysis results will be saved to `/app/output/dewatering_sensitivity_analysis.png` and `/app/output/dewatering_sensitivity_results.xlsx` (if --export-excel is used).

### 5. Customizing Parameters

You can customize the analysis by passing arguments:
```bash
docker run --rm -v $(pwd)/output:/app/output dewatering-tool --dry-tons 25000 --cake-ts-target 22.0 --export-excel
```

Available parameters:

#### Basic Parameters
- `--cake-ts-target`: Target dewatered cake TS% (default: 21.5)
- `--dry-tons`: Annual dry tons produced (default: 29000)

#### Polymer Parameters
- `--dry-polymer-price`: Dry polymer price per pound (default: $1.77/lb)
- `--emulsion-polymer-price`: Emulsion polymer price per pound (default: $2.61/lb)

#### Transportation Parameters
- `--avg-trip-miles`: Average round trip miles (default: 144)
- `--mpg`: Miles per gallon for trucks (default: 6.0)
- `--fuel-price`: Diesel fuel price per gallon (default: $4.00)
- `--max-tons-per-load`: Maximum wet tons per load (default: 21)

#### Step Function Parameters
- `--operator-cost`: Cost per additional operator (default: $100,000)
- `--mechanic-cost`: Cost per additional mechanic (default: $100,000)
- `--equipment-cost`: Annual cost per tractor/trailer (default: $25,000)
- `--operator-threshold`: % TS decrease to add an operator (default: 0.01 = 1%)
- `--mechanic-threshold`: % TS decrease to add a mechanic (default: 0.02 = 2%)
- `--equipment-threshold`: % TS decrease to add equipment (default: 0.01 = 1%)

#### Analysis Options
- `--sensitivity`: Perform detailed sensitivity analysis
- `--sensitivity-values`: TS% values for sensitivity analysis (comma-separated, default: 18,20,22,24)

#### Output Options
- `--output-dir`: Directory to save output files (default: /app/output when using Docker)
- `--export-excel`: Export results to Excel

## Running Locally

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Tool

Basic usage:
```bash
python src/wastewater-dewatering-tool.py
```

With custom parameters:
```bash
python src/wastewater-dewatering-tool.py --dry-tons 25000 --cake-ts-target 22.0 --export-excel --output-dir ./output
```

With sensitivity analysis:
```bash
python src/wastewater-dewatering-tool.py --sensitivity --export-excel --output-dir ./output
```

## Project Structure

```
.
├── Dockerfile            # Docker configuration
├── README.md             # This file
├── requirements.txt      # Python dependencies
├── run_dewatering_tool.sh # Convenience script for running with Docker
└── src/
    └── wastewater-dewatering-tool.py  # Main tool implementation
```

## Analysis Outputs

### Standard Analysis

The tool produces:
- A detailed report showing optimal operating points and cost breakdowns
- A figure showing total cost versus dewatered cake %TS with highlighted optimal points
- (Optional) Excel export with detailed results

**Example Output:**
```
=== Whole Cost Dewatering Analysis Report ===
Annual Dry Tons: 29,000
Target TS%: 21.5%

Optimal Operating Points:
  Dry Polymer: 20.5% TS - $5,240,528.00
  Emulsion Polymer: 20.0% TS - $6,244,462.00

Cost Breakdown at Optimal Points:
  Dry Polymer (20.5% TS):
    Wet Tons: 141,463
    Transportation Costs: $1,160,900.00
    Labor Expenses: $2,334,841.00
    Equipment Expenses: $25,000.00
    Total RR&R Expenses: $3,520,741.00
    Polymer Dose: 34.53 lb/ton
    Polymer Cost: $1,690,764.00
    Total Cost: $5,240,528.00
```

### Sensitivity Analysis

The sensitivity analysis produces:
- A detailed comparison of costs at specific TS% values (18%, 20%, 22%, 24% by default)
- Visualization showing cost breakdown at each TS% value
- Analysis of wet tons with percent changes relative to the target
- Polymer dose curves showing how requirements change with cake TS%
- (Optional) Excel export with sensitivity analysis data

## Interpretation of Results

The tool helps identify the optimal dewatered cake TS% that balances:
- RR&R costs (which increase at higher TS%)
- Polymer costs (which decrease at higher TS%)

Based on the Phase I Assessment report, the ideal range for target dewatered cake %TS is between 20% and 22%, which provides the best overall value to the District.

