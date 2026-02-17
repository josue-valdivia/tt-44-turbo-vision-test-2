# Install dependencies (required for Cython to compile keypoints_cy_all.pyx on first run)
pip install cython numpy opencv-python

# Windows: ensure a C compiler is available (e.g. Visual Studio Build Tools)
# Linux: gcc is usually present

# Run keypoints_convert (first run will compile the .pyx and show C build output)
python ./opt_v3/keypoints_convert.py --video-url ./videos/new/26.mp4 --unsorted-json ./miner_responses_no_ordered/26.json

python keypoints_calculate_score.py --video-url ./videos/new/26.mp4 --miner-json ./miner_responses_ordered_optimised/26.json --verbose

python example_miner/scripts/generate_keypoints.py --video-url https://scoredata.me/chunks/6802dc775f3c463fa048da1e0e7df6.mp4 --out lovely_test/json_galileo/keypoints.json



python ./opt_v4/keypoints_convert.py --video-url ./videos/new/26.mp4 --unsorted-json ./miner_responses_no_ordered/26.json