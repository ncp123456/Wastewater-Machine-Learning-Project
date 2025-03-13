#!/bin/bash

# Script to run the Wholistic Dewatering Tool Docker container with various options

# Default values
DRY_TONS=29000
CAKE_TS=21.5
TS_RANGE=""
EXPORT=false
PLOT=false

# Function to display usage information
function show_help {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help                 Show this help message"
    echo "  -d, --dry-tons VALUE       Set dry tons value (default: 29000)"
    echo "  -c, --cake-ts VALUE        Set target cake TS% (default: 21.5)"
    echo "  -t, --ts-range RANGE       Set TS% range to analyze (comma-separated values, e.g., \"18,19,20,21,22,23,24\")"
    echo "  -e, --export               Export results to Excel"
    echo "  -p, --plot                 Generate and save plots of the results"
    echo ""
    echo "Example:"
    echo "  $0 --dry-tons 25000 --cake-ts 22.0 --export --plot"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        -d|--dry-tons)
            DRY_TONS="$2"
            shift 2
            ;;
        -c|--cake-ts)
            CAKE_TS="$2"
            shift 2
            ;;
        -t|--ts-range)
            TS_RANGE="$2"
            shift 2
            ;;
        -e|--export)
            EXPORT=true
            shift
            ;;
        -p|--plot)
            PLOT=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Build the Docker command
DOCKER_CMD="docker run --rm -v $(pwd)/output:/app/output dewatering-tool"

# Add options to the Docker command
DOCKER_CMD="$DOCKER_CMD --dry-tons $DRY_TONS --cake-ts $CAKE_TS"

if [ -n "$TS_RANGE" ]; then
    DOCKER_CMD="$DOCKER_CMD --ts-range \"$TS_RANGE\""
fi

if [ "$EXPORT" = true ]; then
    DOCKER_CMD="$DOCKER_CMD --export"
fi

if [ "$PLOT" = true ]; then
    DOCKER_CMD="$DOCKER_CMD --plot"
fi

# Create output directory if it doesn't exist
mkdir -p output

# Display the command being run
echo "Running: $DOCKER_CMD"
echo ""

# Execute the Docker command
eval $DOCKER_CMD
