#!/bin/bash
THREADS=(4 16 32)
SIZE=(65536 131072)
for thread in ${THREADS[@]}; 
	do
		export OMP_NUM_THREADS=$thread
		echo "Num threads: $thread"
	for i in ${SIZE[@]}; 
		do 
			./nbody $i
		done
done
