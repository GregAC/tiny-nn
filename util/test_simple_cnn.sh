#!/bin/bash
# End-to-end test script for TNN controller with SimpleCNN model
#
# This script:
# 1. Starts the TNN model server
# 2. Runs the controller with the SimpleCNN model
# 3. Compares output with PyTorch reference
# 4. Reports success/failure
#
# Usage:
#   ./util/test_simple_cnn.sh

set -e

# Configuration
PORT=9876
MODEL_PATH="controller/examples/simple_cnn.toml"
INPUT_PATH="controller/examples/test_input_8x8.json"
OUTPUT_PATH="/tmp/claude/tnn_output.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "TNN SimpleCNN End-to-End Test"
echo "=========================================="
echo

# Check that files exist
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model file not found: $MODEL_PATH${NC}"
    exit 1
fi

if [ ! -f "$INPUT_PATH" ]; then
    echo -e "${RED}Error: Input file not found: $INPUT_PATH${NC}"
    exit 1
fi

# Create output directory
mkdir -p /tmp/claude

# Build everything first
echo -e "${YELLOW}Building model and controller...${NC}"
cargo build -p model -p controller 2>&1 | head -20

echo

# Start model server in background
echo -e "${YELLOW}Starting TNN model server on port $PORT...${NC}"
cargo run -p model -- serve --port $PORT &
SERVER_PID=$!

# Give server time to start
sleep 2

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo -e "${RED}Error: Model server failed to start${NC}"
    exit 1
fi

echo "Server started with PID $SERVER_PID"
echo

# Run controller
echo -e "${YELLOW}Running controller...${NC}"
set +e  # Don't exit on error, we need to cleanup
cargo run -p controller -- run \
    --model "$MODEL_PATH" \
    --input "$INPUT_PATH" \
    --output "$OUTPUT_PATH" \
    --host "localhost:$PORT" \
    --intermediate
CONTROLLER_EXIT=$?
set -e

echo

# Stop server
echo -e "${YELLOW}Stopping server...${NC}"
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo

# Check controller exit status
if [ $CONTROLLER_EXIT -ne 0 ]; then
    echo -e "${RED}Controller failed with exit code $CONTROLLER_EXIT${NC}"
    exit 1
fi

# Check output exists
if [ ! -f "$OUTPUT_PATH" ]; then
    echo -e "${RED}Error: Output file not created: $OUTPUT_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}TNN output:${NC}"
cat "$OUTPUT_PATH"
echo
echo

# Compare with PyTorch (if available)
if command -v python3 &> /dev/null && python3 -c "import torch, toml" 2>/dev/null; then
    echo -e "${YELLOW}Comparing with PyTorch reference...${NC}"
    python3 util/pytorch_compare.py \
        "$MODEL_PATH" \
        "$INPUT_PATH" \
        "$OUTPUT_PATH" \
        --tolerance 0.05 \
        --verbose

    if [ $? -eq 0 ]; then
        echo
        echo -e "${GREEN}=========================================="
        echo "TEST PASSED"
        echo -e "==========================================${NC}"
    else
        echo
        echo -e "${RED}=========================================="
        echo "TEST FAILED - Output mismatch"
        echo -e "==========================================${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}PyTorch not available, skipping comparison${NC}"
    echo "Install with: pip install torch toml"
    echo
    echo -e "${YELLOW}=========================================="
    echo "TEST INCOMPLETE - Manual verification needed"
    echo -e "==========================================${NC}"
fi
