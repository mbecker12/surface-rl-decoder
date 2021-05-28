#!/bin/bash

counter=1
max_counter=10

until [ $counter -gt $max_counter ]
do
  echo Counter: $counter / $max_counter
  python analysis/compare_base_system.py
  ((counter++))
done
