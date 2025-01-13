#!/bin/bash

for i in *.zip
do
mkdir ./${i/.zip//}
unzip $i -d ./${i/.zip//}
done
