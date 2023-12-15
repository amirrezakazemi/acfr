#!/bin/bash
for dataset in "news" "tcga"
do
    sbatch run.sh $dataset
done
