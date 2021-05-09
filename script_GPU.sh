#!/bin/sh


rm -rf plots/data/perf_data.csv

for i in $(seq 1 5)
do
  echo $i
  for tileX in $(seq 1 6)
  do
    for tileY in $(seq 1 6)
    do
      TILEX=$((2**$tileX)) TILEY=$((2**$tileY)) OMP_NUM_THREADS=46 OMP_PLACES=cores ./run -k life -o -v ocl_hybrid -n -i 1000 -a meta3x3 -s 6208 -tw $((2**$tileX)) -th $((2**$tileY))
    done
  done
done
 mv plots/data/perf_data.csv perf_meta_hybrid_tile.csv

 #### calcul random
 rm -rf plots/data/perf_data.csv

for i in $(seq 1 5)
do
  echo $i
  for tileX in $(seq 1 9)
  do
    for tileY in $(seq 1 9)
    do
      TILEX=$((2**$tileX)) TILEY=$((2**$tileY)) OMP_NUM_THREADS=46 OMP_PLACES=cores ./run -k life -o -v ocl_hybrid -n -i 1000 -a random -s 4096 -tw $((2**$tileX)) -th $((2**$tileY))
    done
  done
done
 mv plots/data/perf_data.csv perf_random_hybrid_tile.csv

  #### calcul guns
 rm -rf plots/data/perf_data.csv

for i in $(seq 1 5)
do
  echo $i
  for tileX in $(seq 1 9)
  do
    for tileY in $(seq 1 9)
    do
      TILEX=$((2**$tileX)) TILEY=$((2**$tileY)) OMP_NUM_THREADS=46 OMP_PLACES=cores ./run -k life -o -v ocl_hybrid -n -i 1000 -a guns -s 4096 -tw $((2**$tileX)) -th $((2**$tileY))
    done
  done
done
 mv plots/data/perf_data.csv perf_guns_hybrid_tile.csv

   #### calcul sparse
 rm -rf plots/data/perf_data.csv

for i in $(seq 1 5)
do
  echo $i
  for tileX in $(seq 1 9)
  do
    for tileY in $(seq 1 9)
    do
      TILEX=$((2**$tileX)) TILEY=$((2**$tileY)) OMP_NUM_THREADS=46 OMP_PLACES=cores ./run -k life -o -v ocl_hybrid -n -i 1000 -a sparse -s 4096 -tw $((2**$tileX)) -th $((2**$tileY))
    done
  done
done
 mv plots/data/perf_data.csv perf_sparse_hybrid_tile.csv