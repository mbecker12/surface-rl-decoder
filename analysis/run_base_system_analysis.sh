#!/bin/bash

counter=1
max_counter=50

until [ $counter -gt $max_counter ]
do
  echo Counter: $counter / $max_counter
  python src/analysis/compare_base_system.py
  ((counter++))
done
