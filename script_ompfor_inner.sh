#!/bin/sh
##### PERF opti for de base
rm -f plots/data/perf_data.csv
for i in $(seq 1 46); do
for j in $(seq 1 7); do
echo $i; 
OMP_NUM_THREADS=$i OMP_PLACES=cores ./run -k life -n -i 50 -a random -s 2048 -v tiled_omp_for_inner -th $((2**$j)) -tw $((2**$j)); done; done;
./plots/easyplot.py
mv plot.pdf inner_for.pdf

##### PERF collapse 
rm -f plots/data/perf_data.csv
for i in $(seq 1 46); do
for j in $(seq 1 7); do
echo $i;
OMP_NUM_THREADS=$i OMP_PLACES=cores ./run -k life -n -i 50 -a random -s 2048 -v tiled_omp_for_inner_c -th $((2**$j)) -tw $((2**$j));done; done;
./plots/easyplot.py
mv plot.pdf inner_for_c.pdf

##### PERF collapse dynamic
rm -f plots/data/perf_data.csv
for i in $(seq 1 46); do
for j in $(seq 1 7); do
echo $i;
OMP_NUM_THREADS=$i OMP_PLACES=cores ./run -k life -n -i 50 -a random -s 2048 -v tiled_omp_for_inner_cd -th $((2**$j)) -tw $((2**$j));done; done;
./plots/easyplot.py
mv plot.pdf inner_for_cd.pdf

##### PERF collapse static
rm -f plots/data/perf_data.csv
for i in $(seq 1 46); do
for j in $(seq 1 7); do
echo $i;
OMP_NUM_THREADS=$i OMP_PLACES=cores ./run -k life -n -i 50 -a random -s 2048 -v tiled_omp_for_inner_cs -th $((2**$j)) -tw $((2**$j));done; done;
./plots/easyplot.py
mv plot.pdf inner_for_cs.pdf

##### PERF collapse static 1
rm -f plots/data/perf_data.csv
for i in $(seq 1 46); do
for j in $(seq 1 7); do
echo $i;
OMP_NUM_THREADS=$i OMP_PLACES=cores ./run -k life -n -i 50 -a random -s 2048 -v tiled_omp_for_inner_cs1 -th $((2**$j)) -tw $((2**$j));done; done;
./plots/easyplot.py
mv plot.pdf inner_for_cs1.pdf
