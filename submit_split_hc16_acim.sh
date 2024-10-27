#!/bin/bash
#SBATCH --array=0-19
#SBATCH --account=def-jeromew
#SBATCH --mem=300G                # memory (per node)
#SBATCH --time=03-00:00            # time (DD-HH:MM)

module --force purge
module load StdEnv/2020 python/3.11 scipy-stack/2023b
virtualenv --no-download $SLURM_TMPDIR/nanopore_env2
source $SLURM_TMPDIR/nanopore_env2/bin/activate
pip install pip --upgrade
pip install $HOME/kmodes-0.12.2-py2.py3-none-any.whl  seaborn --no-index
pip install statsmodels --no-index
pip install torch --no-index
pip install --no-index sklearn
pip install numpy --no-index
python3 decon_split_hc16_acim.py $SLURM_ARRAY_TASK_ID 
