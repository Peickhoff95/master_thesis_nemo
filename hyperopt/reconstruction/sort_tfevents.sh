#!/bin/bash
i=0
for f in $1/events.*
do
	echo $f
	mkdir $1/$i
	mv $f $1/$i/
	i=$((i+1))
done
