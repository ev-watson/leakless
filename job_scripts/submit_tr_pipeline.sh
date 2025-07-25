#!/bin/bash
# -e: exit on any error
# -u: treat unset variables as errors
# -o pipefail: propagate failure through pipelines
set -euo pipefail

# Submit the training job and extract its job ID
# awk executes comand in {} which in this case prints the 4th string from the piped output
main_jid=$(sbatch run_train.sh | awk '{print $4}')
echo "Submitted training job with Job ID: $main_jid"

# Expand the wildcard to get the checkpoint filename
# 2>/dev/null redirects the stderr (file descriptor 2) to dev/null
# head grabs the first result from this output
ckpt_path=$(ls tlogs/lowell/checkpoints/epoch*.ckpt 2>/dev/null | head -n 1)

# -z flag checks if string is empty, meaning output is true iff string is empty
if [[ -z "$ckpt_path" ]]; then
    echo "No matching checkpoint file found at tlogs/lowell/checkpoints/epoch*.ckpt"
    exit 1
fi

echo "Will pass checkpoint file: $ckpt_path to run_results.sh"

# Submit the results job with dependency and checkpoint path
sbatch --dependency=afterok:"$main_jid" run_results.sh "$ckpt_path"