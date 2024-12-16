#/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
cd /home/code/tvlab

echo "*** compile start ***"
git reset --hard HEAD
git pull origin master

dst_path=(/home/release/tvlab)
script=(run_full.sh)
chmod u+x tests/${script}

version=$(python -c "exec(open('./tvlab/version.py').read());print(__version__)")
echo "tvlab version: "${version}

function compile(){
    echo " >> compile in env $1"
    # compile
    python setup.py bdist_wheel BSO

    # install
    pkg_name="tvlab-"${version}"-"$1"-"$1"m-linux_x86_64.whl"
    pkg_path="dist/"${pkg_name}
    if [ -e ${pkg_path} ]
    then
        pip install ${pkg_path}
    else
        python setup.py develop
    fi

    # unittest
    if [ $1 == "cp37" ]
    then
        cd tests && mkdir -p log
        echo " >>>> unittest ... "
        out_path="./log/unittest_OK_tvlab-"${version}"-"$1"-"$1"m-linux_x86_64.log"
        echo "unittest log file: "$out_path

        ./${script} &> $out_path
        is_fail=$(cat $out_path | grep -i 'fail*\|error*' )
        if [ -n "$is_fail" ]
        then
            echo "*** unittest FAIL ***"${is_fail}
            dst_file="unittest_FAIL_tvlab-"${version}"-"$1"-"$1"m-linux_x86_64.log"
            mv $out_path log/$dst_file #rename
            mkdir -p ${dst_path}/log
            cp log/$dst_file $dst_path/log/$dst_file
        else
            echo "*** unittest all OK ***"
            # if unittest OK, copy all 35 36 37 tvlab
            cp ../dist/tvlab-${version}-*.whl ${dst_path}
        fi
        cd ../
    fi
    pip uninstall -y tvlab
}

# order 35, 36, 37. should not change
conda activate env_35
compile "cp35"
conda deactivate

conda activate env_36
compile "cp36"
conda deactivate

conda activate env_compile
compile "cp37"
conda deactivate

chmod u-x tests/${script}
echo "*** compile finished ***"
