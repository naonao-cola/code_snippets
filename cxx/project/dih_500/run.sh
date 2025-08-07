#!/bin/sh
chmod 777 app_local_test
export LD_LIBRARY_PATH=./app/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=./app/cl:$LD_LIBRARY_PATH
./app_local_test
