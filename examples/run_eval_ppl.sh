#!/bin/bash

# Call this script to help run multiple iterations of perplexity evaluation.
# Comment out the line for the desired cache.

# List of integers to loop through
values=(50 100 125 250 500 1000 2000 4000)

# Iterate through each value
for i in "${values[@]}"
do
  echo "Running with i=$i"
  CUDA_VISIBLE_DEVICES=5 python original_eval_ppl.py --num_eval_tokens ${i} >> "tmp_${i}.log"
  # CUDA_VISIBLE_DEVICES=5 python koala_eval_ppl.py --num_eval_tokens ${i} >> "tmp_${i}.log"
done
