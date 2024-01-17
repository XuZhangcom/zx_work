#!/bin/bash

d='data'
unzip_target='image'

echo "----unzip file----" $d  ' ...'

if [ ! -d $unzip_target ]; then 
mkdir -p $unzip_target 
fi

cd $d
echo "cd" $(pwd) "..."

for file in `ls *.gz`
do
    echo "unzip" $file "..."
    gzip -dk $file -c > "/home/zx_work/$unzip_target"/${file%%.*}
done
