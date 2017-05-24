#!/bing/bash

old=45779329/

dir_=classification/
if [ ! -d "$dir_" ]; then
    mkdir $dir_
fi
for i in `seq 0 19`; do
    cp $old$i.out $dir_
done
cp transform.py $dir_
cd $dir_
python transform.py classification
cd ..

dir_=regression/
if [ ! -d "$dir_" ]; then
    mkdir $dir_
fi
for i in `seq 0 19`; do
    cp $old$i.out $dir_
done
cp transform.py $dir_
cd $dir_
python transform.py regression
cd ..