#!/bin/bash
#SBATCH --job-name=tada-stream-test
#SBATCH --partition=eval
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/weka/sharath/projects/tada/tests/integration_test.log

cd /mnt/weka/sharath/projects/tada
export PATH="/mnt/weka/sharath/anaconda3/envs/media-pipeline/bin:$PATH"

echo "=== Running integration tests ==="
python -m pytest tests/test_streaming.py -m integration -v -s 2>&1

echo "=== Done ==="
echo "Output files:"
ls -la tests/output/ 2>/dev/null
