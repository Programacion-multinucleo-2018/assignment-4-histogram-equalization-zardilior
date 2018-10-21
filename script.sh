#!/bin/bash
for i in $( ls Images); do
    echo $i
    ./bin/program "Images/$i" 3 >> README.md
done
