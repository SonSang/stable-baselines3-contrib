#!/bin/bash
NUM_PROCS=4

for ((c=0; c<$NUM_PROCS; c++))
do
    OMP_NUM_THREADS=1 nohup python compare_ppo.py --env Swimmer --n_steps 1000 --n_eval_steps 1000 --pe --no-record & sleep 3
done