#!/bin/sh
chmod 777 test04
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH



./test04
