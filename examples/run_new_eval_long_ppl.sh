#!/bin/bash

# List of integers to loop through
# values=(50 100 125 250 500 1000 2000 4000)
values=(1000 2000 4000)

# Iterate through each value
for i in "${values[@]}"
do
  echo "Running with i=$i"
  CUDA_VISIBLE_DEVICES=5 python new_eval_long_ppl.py --num_eval_tokens ${i} >> "tmp_${i}.log"
  # CUDA_VISIBLE_DEVICES=5 python final_newest_llama_eval_long.py --num_eval_tokens ${i} >> "tmp_${i}.log"
done
