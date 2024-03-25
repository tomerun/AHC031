#!/bin/bash -eu

START=$1
END=$2
# mkdir out/_$D/
for (( i = $START; i <= $END; i++ )); do
	seed=$(printf "%04d" $i)
	echo "seed:${seed}" 1>&2
	./main < tools/in/"$seed".txt 2>&1 > out/"$seed".txt | tail -n 1
	tools/target/release/vis tools/in/"$seed".txt out/"$seed".txt
done
