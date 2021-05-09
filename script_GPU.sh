#!/bin/sh


rm -rf plots/data/perf_data.csv

for i in $(seq 1 1)
do
  echo $i
  for tileX in $(seq 0 10)
  do
    for tileY in $(seq 0 10)
    do
      TILEX=$((2**$tileX)) TILEY=$((2**$tileY)) ./run -k life -o -v ocl_finish -n -i 1000 -a meta3x3 -s 6208 -tw $((2**$tileX)) -th $((2**$tileY)) 
    done
  done
done
