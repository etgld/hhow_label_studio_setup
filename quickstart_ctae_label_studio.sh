#!/bin/bash -i
HTTP_PORT=8080 # Label Studio's default
HTTPS_PORT=9090 # Our preferred option
CTAE_PROJECT_DIR=$PWD/projects/rt_ctae
CTAE_DB_DIR=$PWD/db/rt_ctae
LS_HOST=https://bitterman-lab-annotation.net:9090
sh run_label_studio.sh "$HTTPS_PORT" "$CTAE_DB_DIR" "$CTAE_PROJECT_DIR" "$LS_HOST"
