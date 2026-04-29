# Source this file from inside SLURM jobs to load modules and activate the venv.
#   source euler/activate.sh
#
# Edit VENV_DIR below if your venv is elsewhere.

VENV_DIR="${VENV_DIR:-$HOME/venvs/shellsim}"

module load stack/2024-06
module load python_cuda/3.11.6
module load eth_proxy
module load texlive            # for matplotlib rcParams['text.usetex']=True

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
