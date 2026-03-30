#!/usr/bin/env bash

INPUT_TXT="$1"
ANNOTATION_PROGRESS_SAVE_FOLDER="$2"

if [ -z "$INPUT_TXT" ] || [ -z "$ANNOTATION_PROGRESS_SAVE_FOLDER" ]; then
	echo "Usage: $0 <input_txt_path> <annotation_progress_save_folder>"
	exit 1
fi

python mirror3d/annotation/plane_annotation/plane_annotation_tool.py \
 --function 5 \
 --annotation_progress_save_folder "$ANNOTATION_PROGRESS_SAVE_FOLDER" \
 --input_txt "$INPUT_TXT"