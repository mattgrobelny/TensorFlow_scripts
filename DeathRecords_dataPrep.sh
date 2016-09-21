#!/usr/bin

# Prep a training and test data set from DeathRecords for SVM test

# Pull out:
# Id	Sex	Age	MaritalStatus	MannerOfDeath

# and shuffle the values
cat DeathRecords.csv | tr "," "\t" | sed 1d | cut -f 1,7,9,16,20 | shuf > DeathRecords_shuf.tsv

# store last 10k as training and first 10k as test datasets
cat DeathRecords_shuf.tsv | tail -n 10000 > DeathRecords_10k_training.tsv

cat DeathRecords_shuf.tsv | head -n 10000 > DeathRecords_10k_test.tsv
