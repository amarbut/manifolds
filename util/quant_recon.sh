#!/bin/bash

START=/media/anna/Samsung_T5/manifolds/VM_results/
END=/test/metric_results/prajjwal1

for VM in VM_1 VM_2 VM_3 VM_5 VM_6 VM_7 VM_8 VM_9 VM_10
do
echo $VM
  nohup python3 -u  run_reconstruction_skew.py \
    --folder_loc ${START}${VM}${END} 

done
