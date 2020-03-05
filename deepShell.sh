#!/bin/bash

#python auto_diagnosis.py medNorm
#wait
#python auto_diagnosis.py dilantin
#wait
#python auto_diagnosis.py keppra
#
#python auto_diagnosis.py highAge
#wait
#python auto_diagnosis.py ageDiff
#wait
#python auto_diagnosis.py genderNorm
#wait
#python auto_diagnosis.py lowAgeNorm
#wait
#python auto_diagnosis.py midAgeNorm
#wait
#python auto_diagnosis.py highAgeNorm
#wait
#python auto_diagnosis.py ageDiffNorm

for (( t=1; t<=10; t++ )); do
	python auto_diagnosis.py 'dilantin','keppra',0,${t}
	wait
done

for (( t=1; t<=10; t++ )); do
	python auto_diagnosis.py 'dilantin','keppra',1,${t}
	wait
done

for (( t=1; t<=10; t++ )); do
	python auto_diagnosis.py 'none','dilantin',0,${t}
	wait
done

for (( t=1; t<=10; t++ )); do
	python auto_diagnosis.py 'none','keppra',0,${t}
	wait
done

for (( t=1; t<=10; t++ )); do
	python auto_diagnosis.py 'none','dilantin',1,${t}
	wait
done

for (( t=1; t<=10; t++ )); do
	python auto_diagnosis.py 'none','keppra',1,${t}
	wait
done

echo 'Finished all simulation'