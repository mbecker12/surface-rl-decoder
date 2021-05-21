#!/bin/bash

counter=1
max_counter=60

until [ $counter -gt $max_counter ]
do
  echo Counter: $counter / $max_counter
  python analysis/analyze_stack_depth.py
  ((counter++))
done
