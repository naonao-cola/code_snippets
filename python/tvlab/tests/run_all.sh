#!/bin/bash

mkdir -p log/ && rm log/* -rf

./run_tvlab_common_test.sh &> log/tvlab_common_test.log
cat log/tvlab_common_test.log | grep -i 'fail*\|error*'

./run_tvdl_test.sh &> log/tvdl_test.log
cat log/tvdl_test.log | grep -i 'fail*\|error*'

./run_tvlab_cv_test.sh &> log/tvlab_cv_test.log
cat log/tvlab_cv_test.log | grep -i 'fail*\|error*'

# tail -f log/*.log -n 500

# is_fail=$(cat $log_path | grep -i 'fail*\|error*' )
# if [ -n "$is_fail" ]
# then
#     echo "*** FAIL unittest ${pkg_name} ***"
#     echo "see detail, go to log/*.log"
# else
#     echo "*** SUCCESS unittest ${pkg_name} ***"
#     echo "see detail, go to log/*.log"
# fi