#!/bin/bash

# Ariel Data Challenge Submission Runner Script
# Usage: ./submission/run_submission.sh [config_name] [options]

set -e  # Exit on any error

# Default values
CONFIG_NAME="baseline_default"
VALIDATE=true
EVALUATE=false
GROUND_TRUTH=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print usage
usage() {
    echo "Usage: $0 [config_name] [options]"
    echo ""
    echo "Config names:"
    echo "  baseline_default     - Standard baseline pipeline"
    echo "  baseline_high_precision - High precision baseline"
    echo "  bayesian_default     - Bayesian pipeline with MCMC"
    echo ""
    echo "Options:"
    echo "  --no-validate        - Skip format validation"
    echo "  --evaluate           - Evaluate submission (requires --ground-truth)"
    echo "  --ground-truth FILE  - Ground truth file for evaluation"
    echo "  --help              - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run default baseline"
    echo "  $0 baseline_high_precision            # Run high precision baseline"
    echo "  $0 bayesian_default --no-validate    # Run Bayesian without validation"
    echo "  $0 baseline_default --evaluate --ground-truth dataset/test_ground_truth.csv"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-validate)
            VALIDATE=false
            shift
            ;;
        --evaluate)
            EVALUATE=true
            shift
            ;;
        --ground-truth)
            GROUND_TRUTH="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        -*)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
        *)
            CONFIG_NAME="$1"
            shift
            ;;
    esac
done

# Check if config file exists
CONFIG_FILE="submission/configs/${CONFIG_NAME}.yml"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo -e "${RED}Error: Configuration file not found: $CONFIG_FILE${NC}"
    echo "Available configurations:"
    ls submission/configs/*.yml 2>/dev/null | sed 's/.*\//  /' | sed 's/\.yml$//' || echo "  No configuration files found"
    exit 1
fi

echo -e "${GREEN}=== Ariel Data Challenge Submission Runner ===${NC}"
echo "Configuration: $CONFIG_NAME"
echo "Config file: $CONFIG_FILE"
echo "Validation: $VALIDATE"
echo "Evaluation: $EVALUATE"
echo ""

# Build command
CMD="python submission/submission_runner.py $CONFIG_FILE"
if [[ "$VALIDATE" == true ]]; then
    CMD="$CMD --validate"
fi

# Run submission
echo -e "${YELLOW}Running submission...${NC}"
echo "Command: $CMD"
echo ""

# Execute the command and capture the output
if OUTPUT=$(eval $CMD 2>&1); then
    echo "$OUTPUT"
    
    # Extract submission file path from output
    SUBMISSION_FILE=$(echo "$OUTPUT" | grep "Submission saved to:" | sed 's/.*Submission saved to: //')
    
    if [[ -n "$SUBMISSION_FILE" && -f "$SUBMISSION_FILE" ]]; then
        echo -e "${GREEN}✓ Submission completed successfully!${NC}"
        echo "Submission file: $SUBMISSION_FILE"
        
        # Get file size
        FILE_SIZE=$(du -h "$SUBMISSION_FILE" | cut -f1)
        echo "File size: $FILE_SIZE"
        
        # Get number of planets
        NUM_PLANETS=$(tail -n +2 "$SUBMISSION_FILE" | wc -l)
        echo "Number of planets: $NUM_PLANETS"
        
        # Run evaluation if requested
        if [[ "$EVALUATE" == true ]]; then
            echo ""
            echo -e "${YELLOW}Running evaluation...${NC}"
            
            if [[ -z "$GROUND_TRUTH" ]]; then
                echo -e "${RED}Error: --ground-truth required for evaluation${NC}"
                exit 1
            fi
            
            if [[ ! -f "$GROUND_TRUTH" ]]; then
                echo -e "${RED}Error: Ground truth file not found: $GROUND_TRUTH${NC}"
                exit 1
            fi
            
            EVAL_CMD="python submission/evaluator.py $SUBMISSION_FILE --ground-truth $GROUND_TRUTH"
            echo "Evaluation command: $EVAL_CMD"
            echo ""
            
            if eval $EVAL_CMD; then
                echo -e "${GREEN}✓ Evaluation completed successfully!${NC}"
            else
                echo -e "${RED}✗ Evaluation failed${NC}"
                exit 1
            fi
        fi
        
    else
        echo -e "${RED}✗ Could not find generated submission file${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ Submission failed${NC}"
    echo "$OUTPUT"
    exit 1
fi

echo ""
echo -e "${GREEN}=== Submission Complete ===${NC}"

# Show next steps
echo ""
echo "Next steps:"
echo "1. Check the submission file: $SUBMISSION_FILE"
if [[ "$EVALUATE" == false ]]; then
    echo "2. Evaluate locally: python submission/evaluator.py $SUBMISSION_FILE --ground-truth YOUR_GROUND_TRUTH.csv"
fi
echo "3. Submit to the competition platform"
echo ""
echo "Generated files are saved in: submission/results/" 