#!/bin/bash
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
#
# Bash script to record video of trained policy.
#
# Usage:
#   ./record_policy.sh [OPTIONS]
#
# Examples:
#   # Record with default settings (uses latest checkpoint)
#   ./record_policy.sh
#
#   # Record specific checkpoint with custom duration
#   ./record_policy.sh -c model_11200.pt -d 15
#
#   # Record with custom camera angle
#   ./record_policy.sh -p "3.0,3.0,2.0" -l "0.0,0.0,0.8"
#
#   # Record specific task and run
#   ./record_policy.sh -t lite_run -r 2025-03-24_01-31-26 -o run_demo.mp4

set -e

# Default values
TASK="lite_walk"
LOAD_RUN=""
CHECKPOINT=""
OUTPUT="output.mp4"
DURATION=10
FPS=30
CAMERA_POS="2.5,2.5,1.5"
CAMERA_LOOKAT="0.0,0.0,0.6"
NUM_ENVS=1

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Function to show usage
usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Record video of trained TienKung-Lab policy.

OPTIONS:
    -h, --help              Show this help message
    -t, --task TASK         Task name (default: lite_walk)
    -r, --run RUN_DIR       Training run directory (e.g., 2025-03-24_01-31-26)
    -c, --checkpoint CHECK  Checkpoint file (e.g., model_11200.pt)
    -o, --output PATH       Output video path (default: output.mp4)
    -d, --duration SECONDS  Recording duration (default: 10)
    -f, --fps FPS           Video frame rate (default: 30)
    -p, --pos X,Y,Z         Camera position (default: 2.5,2.5,1.5)
    -l, --lookat X,Y,Z      Camera lookat point (default: 0.0,0.0,0.6)
    -n, --num-envs N        Number of environments (default: 1)

CAMERA PRESETS:
    --side                  Side view (pos: 2.5,0.0,1.0)
    --front                 Front view (pos: 0.0,3.0,1.0)
    --top                   Top view (pos: 0.0,0.0,3.0)
    --iso                   Isometric view (pos: 3.0,3.0,2.0)

EXAMPLES:
    # Basic recording with latest checkpoint
    ./$(basename "$0")

    # Record 20 seconds of model_11200.pt
    ./$(basename "$0") -c model_11200.pt -d 20

    # Record specific task with side view
    ./$(basename "$0") -t lite_run --side -o run_side.mp4

    # Record with custom camera
    ./$(basename "$0") -p "4.0,4.0,3.0" -l "0.0,0.0,1.0" -d 30

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -t|--task)
            TASK="$2"
            shift 2
            ;;
        -r|--run)
            LOAD_RUN="$2"
            shift 2
            ;;
        -c|--checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT="$2"
            shift 2
            ;;
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        -f|--fps)
            FPS="$2"
            shift 2
            ;;
        -p|--pos)
            CAMERA_POS="$2"
            shift 2
            ;;
        -l|--lookat)
            CAMERA_LOOKAT="$2"
            shift 2
            ;;
        -n|--num-envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --side)
            CAMERA_POS="2.5,0.0,1.0"
            CAMERA_LOOKAT="0.0,0.0,0.8"
            shift
            ;;
        --front)
            CAMERA_POS="0.0,3.0,1.0"
            CAMERA_LOOKAT="0.0,0.0,0.8"
            shift
            ;;
        --top)
            CAMERA_POS="0.0,0.0,3.0"
            CAMERA_LOOKAT="0.0,0.0,0.0"
            shift
            ;;
        --iso)
            CAMERA_POS="3.0,3.0,2.0"
            CAMERA_LOOKAT="0.0,0.0,0.8"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Build command
cmd="python ${PROJECT_DIR}/legged_lab/scripts/play_and_record.py"
cmd+=" --task=${TASK}"
cmd+=" --video_path=${OUTPUT}"
cmd+=" --video_duration=${DURATION}"
cmd+=" --video_fps=${FPS}"
cmd+=" --camera_pos=${CAMERA_POS}"
cmd+=" --camera_lookat=${CAMERA_LOOKAT}"
cmd+=" --num_envs=${NUM_ENVS}"

if [[ -n "$LOAD_RUN" ]]; then
    cmd+=" --load_run=${LOAD_RUN}"
fi

if [[ -n "$CHECKPOINT" ]]; then
    cmd+=" --load_checkpoint=${CHECKPOINT}"
fi

# Print configuration
echo "========================================"
echo "  TienKung-Lab Policy Video Recorder"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Task:        ${TASK}"
echo "  Run:         ${LOAD_RUN:-<latest>}"
echo "  Checkpoint:  ${CHECKPOINT:-<latest>}"
echo "  Output:      ${OUTPUT}"
echo "  Duration:    ${DURATION}s"
echo "  FPS:         ${FPS}"
echo "  Camera Pos:  ${CAMERA_POS}"
echo "  Camera Look: ${CAMERA_LOOKAT}"
echo "  Num Envs:    ${NUM_ENVS}"
echo ""
echo "Command:"
echo "  ${cmd}"
echo ""
echo "========================================"
echo ""

# Check if in tmux or have display
if [[ -z "$DISPLAY" && -z "${TMUX:-}" ]]; then
    echo "WARNING: No DISPLAY environment variable set."
    echo "This script requires a display for viewport rendering."
    echo "Please run within a tmux session with display access."
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if output directory exists
OUTPUT_DIR=$(dirname "$OUTPUT")
if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "Creating output directory: ${OUTPUT_DIR}"
    mkdir -p "$OUTPUT_DIR"
fi

# Run the recording
echo "Starting recording..."
echo ""
cd "$PROJECT_DIR"
eval "$cmd"

# Check if video was created
if [[ -f "$OUTPUT" ]]; then
    echo ""
    echo "========================================"
    echo "  Recording Complete!"
    echo "========================================"
    echo ""
    echo "Video saved to: $(realpath "$OUTPUT")"
    echo ""
    # Get video info
    if command -v ffprobe &> /dev/null; then
        echo "Video info:"
        ffprobe -v error -select_streams v:0 -show_entries stream=width,height,duration,r_frame_rate -of csv=p=0 "$OUTPUT" 2>/dev/null || true
    fi
else
    echo ""
    echo "ERROR: Video file was not created!"
    exit 1
fi
