#!/bin/bash
cmd /c "taskkill /F /IM blender.exe"
python Crime-Investigation-AI/tests/test_render_pipeline.py &
PID=$!
echo "Started test with PID $PID"

MAX_WAIT=600
COUNT=0
while [ $COUNT -lt $MAX_WAIT ]; do
    if ! ps -p $PID > /dev/null; then
        echo "Process finished."
        break
    fi
    if [ -f Crime-Investigation-AI/tests/animation_test_data/test_output.mp4 ]; then
        echo "Video detected!"
        break
    fi
    sleep 10
    COUNT=$((COUNT + 10))
    echo "Waiting... $COUNT"
done

if [ -f Crime-Investigation-AI/tests/animation_test_data/test_output.mp4 ]; then
    echo "SUCCESS: Video found."
    ls -l Crime-Investigation-AI/tests/animation_test_data/test_output.mp4
else
    echo "FAILED: Video not found."
fi
