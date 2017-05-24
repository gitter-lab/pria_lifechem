#!/bing/bash

declare -i count=0
for i in `seq 0 4`; do
    for j in `seq 0 3`; do
        # echo result_2017_05_07_17_5fold_test$i\_$j.txt
        cp result_2017_05_07_17_5fold_test$i\_$j.txt $count.out
        count=count+1
    done
done
