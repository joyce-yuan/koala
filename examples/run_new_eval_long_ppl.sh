#!/bin/bash

# List of integers to loop through
values=( 100 125 250 500)

# Iterate through each value
for i in "${values[@]}"
do
  echo "Running with i=$i"
  CUDA_VISIBLE_DEVICES=6 python new_eval_long_ppl.py --num_eval_tokens ${i} >> "tmp_${i}.log"
done
