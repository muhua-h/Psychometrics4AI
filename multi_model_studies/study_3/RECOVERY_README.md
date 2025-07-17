# Study 3 Participant Recovery Script

This directory contains `recover_missing_participants.py`, a script that can detect and recover missing participants across all Study 3 formats (Binary, Expanded, Likert).

## Overview

The recovery script:
1. **Detects missing participants** by analyzing simulation results and identifying problematic responses
2. **Validates responses** using the same logic as `unified_convergent_analysis.py`
3. **Recovers missing data** by re-running simulations for problematic participants
4. **Supports all formats** with format-specific data preparation and validation

## Key Differences from Study 2 Recovery Script

### Data Source
- **Study 2**: Uses empirical BFI-2 data from `study2_preprocessed_data.csv`
- **Study 3**: Uses statistically simulated BFI-2 data from `facet_lvl_simulated_data.csv`

### Format Support
- **Study 2**: Binary (Simple/Elaborated), Expanded (I am/You are), Likert
- **Study 3**: Binary, Expanded, Likert (unified approach)

### Data Preparation
- **Study 2**: Simple conversion from preprocessed data
- **Study 3**: Full format conversion using `expanded_scale` and `likert_scale` mappings from `schema_bfi2.py`

### File Patterns
- **Study 2**: Complex patterns with multiple subdirectories
- **Study 3**: Clean patterns in unified result directories

## Usage

### Basic Usage

```bash
# Analyze all formats (dry run - no actual recovery)
python recover_missing_participants.py --dry-run

# Recover missing participants across all formats
python recover_missing_participants.py

# Analyze specific format only
python recover_missing_participants.py --dry-run --format expanded
python recover_missing_participants.py --dry-run --format binary
python recover_missing_participants.py --dry-run --format likert

# Recover missing participants for specific format
python recover_missing_participants.py --format expanded
```

### Options

- `--dry-run`: Analyze missing participants without performing recovery
- `--format {binary,expanded,likert,all}`: Target specific format (default: all)

## How It Works

### 1. Analysis Phase
For each format and model combination:
- Loads simulation results from JSON files
- Validates each response using Mini-Marker trait mapping
- Checks for missing traits, invalid values, and parsing errors
- Calculates how many participants survive aggregation and filtering

### 2. Recovery Phase (if not dry-run)
For each problematic participant:
- Loads original BFI-2 data from `facet_lvl_simulated_data.csv`
- Converts to appropriate format description:
  - **Binary**: Uses `generate_binary_personality_description()` with threshold=2.5
  - **Expanded**: Uses `expanded_scale` mapping with reverse coding
  - **Likert**: Uses `likert_scale` mapping with numeric scores
- Re-runs simulation using the same model and temperature
- Updates original JSON file with recovered responses
- Creates backup files before making changes

### 3. Validation Logic
Uses the same validation rules as `unified_convergent_analysis.py`:
- **Expected Traits**: All 40 Mini-Marker traits must be present
- **Valid Values**: All values must be integers 1-9
- **Domain Mapping**: Uses `minimarker_domain_mapping` for aggregation
- **Reverse Coding**: Applies `reverse_coded_traits` for proper scoring

## Format-Specific Data Preparation

### Binary Format
```python
# Convert domain scores to binary descriptions
participant_data = {
    'bfi2_e': row['bfi_e'],  # Use domain scores
    'bfi2_a': row['bfi_a'],
    'bfi2_c': row['bfi_c'],
    'bfi2_n': row['bfi_n'],
    'bfi2_o': row['bfi_o']
}
binary_description = generate_binary_personality_description(participant_data)
```

### Expanded Format
```python
# Convert individual BFI items to expanded descriptions
def create_expanded_description(data_row):
    # Apply reverse coding
    # Map to expanded_scale descriptions
    # Combine all 60 item descriptions
    return combined_description
```

### Likert Format  
```python
# Convert individual BFI items to likert descriptions
def create_likert_description(data_row):
    # Use original values (no reverse coding)
    # Map to likert_scale prompts with numeric scores
    return combined_description
```

## File Structure

```
study_3/
├── recover_missing_participants.py         # Main recovery script
├── facet_lvl_simulated_data.csv            # Source data (200 participants)
├── study_3_binary_results/                 # Binary format results
│   └── bfi_to_minimarker_binary_*.json
├── study_3_expanded_results/                # Expanded format results
│   └── bfi_to_minimarker_*.json
├── study_3_likert_results/                  # Likert format results
│   └── bfi_to_minimarker_*.json
└── unified_analysis_results/                # Analysis outputs
```

## Expected Output

### Dry Run Mode
```
============================================================
ANALYZING BINARY FORMAT FOR MISSING PARTICIPANTS
============================================================
Expected participants: 200
Found 4 model files

--- Analyzing gpt_4 ---
Total responses: 200
Valid responses: 200
Problematic responses: 0
After aggregation and filtering: 200 participants
Missing participants: 0

SUMMARY OF MISSING PARTICIPANTS
Binary Format:
  gpt_4: 0 missing participants
  gpt_4o: 0 missing participants
  deepseek: 0 missing participants
  llama: 5 missing participants

Total missing participants across all formats: 5
```

### Recovery Mode
```
============================================================
RECOVERING PARTICIPANTS FOR LLAMA
============================================================
Attempting to recover 5 participants
Using model: llama
Using format type: expanded
Running simulation for 5 participants...
Successfully recovered 4 participants
Creating backup: .../bfi_to_minimarker_llama_temp1.backup.json
Updated results saved to: .../bfi_to_minimarker_llama_temp1.json

RECOVERY COMPLETE
Total participants recovered: 4
Success rate: 80.0%
```

## Troubleshooting

### Common Issues

1. **Data File Not Found**
   ```
   Error: Data file not found at facet_lvl_simulated_data.csv
   ```
   **Solution**: Run `python bfi2_facet_level_parameter_extraction_and_simulation.py` first

2. **No Simulation Results**
   ```
   Warning: No JSON files found matching pattern
   ```
   **Solution**: Run the appropriate simulation notebook first

3. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'schema_bfi2'
   ```
   **Solution**: Ensure you're running from the `study_3/` directory

### Validation Tips

- **Before Recovery**: Run `--dry-run` to see missing participants
- **After Recovery**: Run `python unified_convergent_analysis.py` to verify
- **Backup Files**: Check `.backup.json` files if something goes wrong
- **Format Verification**: Compare with original notebook results

## Integration with Analysis Pipeline

1. **Generate Data**: `python bfi2_facet_level_parameter_extraction_and_simulation.py`
2. **Run Simulations**: Execute notebook(s) for desired format(s)
3. **Check Results**: `python recover_missing_participants.py --dry-run`
4. **Recover Missing**: `python recover_missing_participants.py`
5. **Verify Results**: `python unified_convergent_analysis.py`

## Performance Notes

- **Analysis Speed**: ~30 seconds for all formats
- **Recovery Speed**: Depends on number of missing participants and API response time
- **API Usage**: Only makes calls for problematic participants (efficient)
- **Memory Usage**: Loads full dataset but processes incrementally

## Dependencies

- pandas
- numpy
- json
- pathlib
- argparse
- datetime
- Custom modules from `../shared/`:
  - `simulation_utils`
  - `mini_marker_prompt`
  - `binary_baseline_prompt`
  - `schema_bfi2`

## Next Steps

After recovery, consider:
1. **Re-run Analysis**: `python unified_convergent_analysis.py`
2. **Compare Formats**: Check format-wise performance differences
3. **Model Comparison**: Identify models with consistent recovery needs
4. **Quality Assessment**: Review recovered vs. original response quality 