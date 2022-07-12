#!/bin/bash
NUM_PROCS=31

for ((c=0; c<$NUM_PROCS; c++))
do
    OMP_NUM_THREADS=1 nohup python compare_trpo.py --env Swimmer --n_steps 1000_000 --n_eval_steps 10000 --pe --no-record & sleep 1
done
