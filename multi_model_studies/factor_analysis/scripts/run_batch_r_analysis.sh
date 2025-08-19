#!/bin/bash

# Batch R-based CFA Analysis Script
# Runs comprehensive CFA analysis on all simulation results

SCRIPT_DIR="/Users/mhhuang/Psychometrics4AI_revision/multi_model_studies/factor_analysis/scripts"
R_SCRIPT="$SCRIPT_DIR/cfa_analysis_simple.R"
BASE_DIR="/Users/mhhuang/Psychometrics4AI_revision/multi_model_studies"
RESULTS_DIR="$BASE_DIR/factor_analysis/results_r"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Function to analyze all JSON files in a directory
analyze_study() {
    local study_path="$1"
    local study_name="$2"
    
    echo "=== Processing $study_name ==="
    
    # Find all JSON files
    find "$study_path" -name "*.json" | while read json_file; do
        
        # Determine format from path
        if [[ "$json_file" == *"simple"* ]]; then
            format="binary_simple"
        elif [[ "$json_file" == *"elaborated"* ]]; then
            format="binary_elaborated"
        elif [[ "$json_file" == *"expanded"* ]]; then
            format="expanded"
        elif [[ "$json_file" == *"likert"* ]]; then
            format="likert"
        else
            format="unknown"
        fi
        
        # Create output directory
        output_dir="$RESULTS_DIR/$study_name/${format}_format"
        mkdir -p "$output_dir"
        
        echo "Analyzing: $(basename "$json_file")"
        echo "Study: $study_name, Format: $format"
        
        # Run R analysis
        Rscript "$R_SCRIPT" "$json_file" "$output_dir"
        
        echo "---"
    done
}

# Process each study
echo "Starting comprehensive R-based CFA analysis..."

# Study 2
analyze_study "$BASE_DIR/study_2" "study_2"

# Study 3  
analyze_study "$BASE_DIR/study_3" "study_3"

echo "=== Analysis Complete ==="
echo "Results saved to: $RESULTS_DIR"

# Create summary
echo "Generating comprehensive summary..."
find "$RESULTS_DIR" -name "*_R_factor_analysis.csv" | while read csv_file; do
    echo "Found results: $csv_file"
done

echo "All analyses complete!"