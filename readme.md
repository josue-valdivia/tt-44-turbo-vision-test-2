# Install dependencies (required for Cython to compile keypoints_cy_all.pyx on first run)
pip install cython numpy opencv-python

# Windows: ensure a C compiler is available (e.g. Visual Studio Build Tools)
# Linux: gcc is usually present

# Run keypoints_convert (first run will compile the .pyx and show C build output)
python ./opt_v3/keypoints_convert.py --video-url ./videos/new/28.mp4 --unsorted-json ./miner_responses_no_ordered/28.json
python keypoints_calculate_score.py --video-url ./videos/new/28.mp4 --miner-json ./miner_responses_ordered_optimised/28.json --verbose