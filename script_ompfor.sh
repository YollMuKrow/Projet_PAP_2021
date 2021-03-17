#!/bin/sh
rm -f plots/data/perf_data.csv

OMP_NUM_THREADS=46 OMP_PLACES=cores OMP_NUM_THREAD=1 OMP_PLACES=cores ./run -k life -n -i 50 -a random -s 2048 -v tiled_omp_for -th 16 -tw 2

for i in $(seq 1 7); do
OMP_NUM_THREADS=46 OMP_PLACES=cores OMP_NUM_THREAD=46 OMP_PLACES=cores ./run -k life -n -i 50 -a random -s 2048 -v tiled_omp_for -th $((2**$i)) -tw $((2**$i)); done;

for i in $(seq 1 7); do
OMP_NUM_THREADS=46 OMP_PLACES=cores OMP_NUM_THREAD=46 OMP_PLACES=cores ./run -k life -n -i 50 -a random -s 2048 -v tiled_omp_for_c -th $((2**$i)) -tw $((2**$i));done;

for i in $(seq 1 7); do
OMP_NUM_THREADS=46 OMP_PLACES=cores OMP_NUM_THREAD=46 OMP_PLACES=cores ./run -k life -n -i 50 -a random -s 2048 -v tiled_omp_for_cd -th $((2**$i)) -tw $((2**$i));done;

for i in $(seq 1 7); do
OMP_NUM_THREADS=46 OMP_PLACES=cores OMP_NUM_THREAD=46 OMP_PLACES=cores ./run -k life -n -i 50 -a random -s 2048 -v tiled_omp_for_cs -th $((2**$i)) -tw $((2**$i));done;

for i in $(seq 1 7); do
OMP_NUM_THREADS=46 OMP_PLACES=cores OMP_NUM_THREAD=46 OMP_PLACES=cores ./run -k life -n -i 50 -a random -s 2048 -v tiled_omp_for_cs1 -th $((2**$i)) -tw $((2**$i));done;

./plots/easyplot.py
