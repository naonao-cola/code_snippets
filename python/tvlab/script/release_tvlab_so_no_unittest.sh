#/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
cd /home/code/tvlab

git reset --hard HEAD
git pull origin master

function compile(){
    python setup.py bdist_wheel BSO
}

conda activate env_37
compile
conda deactivate

conda activate env_35
compile
conda deactivate

conda activate env_36
compile
conda deactivate

cp dist/* /home/release/tvlab
