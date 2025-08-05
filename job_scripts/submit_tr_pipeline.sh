#!/bin/bash
# -e: exit on any error
# -u: treat unset variables as errors
# -o pipefail: propagate failure through pipelines
set -euo pipefail

# Submit the training job and extract its job ID
# awk executes comand in {} which in this case prints the 4th string from the piped output
main_jid=$(sbatch run_train.sh | awk '{print $4}')
echo "Submitted training job with Job ID: $main_jid"

# Submit the results job with dependency and checkpoint path
sbatch --dependency=afterok:"$main_jid" run_results.sh --resolve-latest