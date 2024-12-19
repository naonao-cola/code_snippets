#!/bin/bash

root=$(cd $(dirname $0); cd ..; pwd)
cython_root=${root}"/cython/"
so_suffix=".cpython-37m-x86_64-linux-gnu.so"

#build c_et199
# cd ${root}"/c_et199"
# python setup.py build_ext --inplace
# cp ${root}"/c_et199/c_et199.pyd" ${root}/tvlab/utils/impl/
# cd ${root}

#build license
cd ${cython_root}"/license"
python setup.py build_ext --inplace
cp ${cython_root}"/license/license.pyd" ${root}/tvlab/utils/impl/
cp ${cython_root}"/license/license"${so_suffix} ${root}/tvlab/utils/impl/
cd ${root}

declare -A cython_task
cython_task=(
    ["cqr_decode"]=${root}"/tvlab/cv/barcode/qrcode/impl/" \
    ["cshape_based_matching"]=${root}"/tvlab/cv/matching/impl/" \
    ["cdevernay"]=${root}"/tvlab/cv/impl/" \
    ["c_caliper"]=${root}"/tvlab/cv/caliper/impl/")

echo ${!cython_task[*]}
echo ${cython_task[*]}
echo "--------------"
for sName in ${!cython_task[@]};
do
    out_dir=${cython_task[${sName}]}
    echo "name:"$sName
    echo "out_dir:"$out_dir
	build_dir=${cython_root}"/"${sName}
    echo "build_dir:"$build_dir
    cd $build_dir
    python setup.py build_ext --inplace
    cp ${build_dir}"/"${sName}".pyd" ${out_dir}
    cp ${build_dir}"/"${sName}${so_suffix} ${out_dir}
    cd ${root}
done	

