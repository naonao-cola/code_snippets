#/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate

src_dir=(/home/cxj/private/src)
dst_dir=(/home/cxj/shared/pkg)
tvlab_script=(run_tvlab_all_test.sh)
tvdl_script=(run_tvdl_test.sh)


# demo: pull_code "tvlab"
function pull_code(){
    cd ${src_dir}/$1
    git reset --hard HEAD
    git pull origin master
    if [ $1 == 'tvlab' ]
    then
        chmod u+x ./tests/${tvlab_script}
        chmod u+x ./tests/${tvdl_script}
    fi
}


# demo: compile_pkg "tvlab" "py37"
function compile_pkg(){
    py_v=$1
    pkg_name=$2
    cd ${src_dir}/${pkg_name}

    echo "############### $py_v $pkg_name ##################"
    v_file="./${pkg_name}/version.py"
    version=$(python -c "exec(open('${v_file}').read());print(__version__)")
    echo ${pkg_name}" version: "${version}

    # compile
    echo "********** compile $py_v $pkg_name ********"
    python setup.py bdist_wheel BSO

    if [ $py_v == "cp37" ]
    then
        # install
        pkg_file="dist/${pkg_name}-${version}-${py_v}-${py_v}m-linux_x86_64.whl"
        if [ -e ${pkg_file} ]
        then
            pip install ${pkg_file}
        else
            python setup.py develop
        fi

        # test
        test_pkg $py_v $pkg_name

        if [ $pkg_name == 'tvdl' ]
        then
            pip uninstall -y tvlab tvdl
        fi
    fi
}


function test_pkg(){
    py_v=$1
    pkg_name=$2
    if [ $py_v == "cp37" ]
    then
        echo "************ test_pkg $pkg_name ***************"
        cd ${src_dir}/${pkg_name}
        v_file="./${pkg_name}/version.py"
        version=$(python -c "exec(open('${v_file}').read());print(__version__)")

        cd ${src_dir}/tvlab/tests && mkdir -p log
        log_path="./log/${pkg_name}-${version}-$py_v-*-unittest_SUCCESS.log"
        echo " >>>> unittest "${pkg_name}" ... "
        echo "unittest log file -> "$log_path

        eval script='$'${pkg_name}_script
        ./${script} &> $log_path
        is_fail=$(cat $log_path | grep -i 'fail*\|error*' )
        if [ -n "$is_fail" ]
        then
            echo "*** FAIL unittest ${pkg_name} ***"
            echo ${is_fail}
            fail_file="${pkg_name}-${version}-$py_v-*-unittest_FAIL.log"
            mv $log_path ./log/$fail_file #rename
            mkdir -p ${dst_dir}/${pkg_name}/log
            cp ./log/$fail_file ${dst_dir}/${pkg_name}/log/$fail_file
        else
            echo "*** SUCCESS unittest ${pkg_name} ***"
            # if SUCCESS, copy all 35 36 37 tvlab/tvdl
            cp ${src_dir}/${pkg_name}/dist/${pkg_name}-${version}-*.whl ${dst_dir}/${pkg_name}/
        fi
        cd ../
    fi
}

pull_code "tvdl"
pull_code "tvlab"

# order 35, 36, 37. should not change
conda activate env_35
compile_pkg "cp35" "tvlab"
compile_pkg "cp35" "tvdl"
conda deactivate

conda activate env_36
compile_pkg "cp36" "tvlab"
compile_pkg "cp36" "tvdl"
conda deactivate

conda activate env_compile
compile_pkg "cp37" "tvlab"
compile_pkg "cp37" "tvdl"
conda deactivate

echo "############ compile finished ############"