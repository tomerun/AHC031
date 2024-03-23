#!/bin/bash -eu

START=$1
END=$2
# mkdir out/_$D/
for (( i = $START; i <= $END; i++ )); do
	seed=$(printf "%04d" $i)
	echo "seed:${seed}" 1>&2
	./main < in/"$seed".txt > out/"$seed".txt
	tools/target/release/vis in/"$seed".txt out/"$seed".txt
done
