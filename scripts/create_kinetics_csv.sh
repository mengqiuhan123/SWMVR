#!/bin/bash

VIDEO_DIR=" "  
CSV_FILE=" "


echo "Generating CSV file..."
> $CSV_FILE  

count=0
for video in "$VIDEO_DIR"/*.mp4; do
    if [ -f "$video" ]; then
        filename=$(basename "$video")
        echo "/ /$filename,-1" >> $CSV_FILE
    fi
done

echo "Generated $count entries in $CSV_FILE"
echo "First few entries:"
head -5 $CSV_FILE