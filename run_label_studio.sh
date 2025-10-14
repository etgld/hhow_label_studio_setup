#!/bin/bash -i
# Credit to Gaby Dinh
source $PWD/.venv/bin/activate

# Argument variables
PORT=$1
DB_PATH=$2
PROJECT_DIR=$3
LS_HOST=$4

# Check if all required arguments are provided
if [ -z "$PORT" ] || [ -z "$DB_PATH" ] || [ -z "$PROJECT_DIR" ] || [ -z "LS_HOST" ]; then
    echo "Error: Missing arguments."
    echo "Usage: $0 <port> <db_path> <project_dir> <label_studio_host>"
    exit 1
fi

# Start Label Studio
echo "Starting Label Studio on port $PORT with database $DB_PATH and project directory $PROJECT_DIR"
export LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK=true
nohup label-studio start --port $PORT --database $DB_PATH --data-dir $PROJECT_DIR --host $LS_HOST > $PROJECT_DIR/labelstudio_$PORT.log 2>&1 &

echo "Label Studio started at $LS_HOST . Logs are being written to $PROJECT_DIR/labelstudio_$PORT.log"
