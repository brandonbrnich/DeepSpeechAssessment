#!/bin/bash

#this is shell script in order to generate all the results for DeepSpeech and GoogleCloud API
#run this shell script when updates occur to either ASR technologies to get updated results

# basic command: deepspeech --model deepspeech-0.9.3-models.pbmm --scorer deepspeech-0.9.3-models.scorer --audio audio/2830-3980-0043.wav

# need to generate a text outputs for each of audio recordings
for f in ./audio/*.wav; do
    d="${f:8:1}"
    newName="basename $f .wav"
    command=$(eval $newName)
    if [ $d == "1" ]
    then 
        deepspeech --model deepspeech-0.9.3-models.pbmm --scorer deepspeech-0.9.3-models.scorer --audio $f > ./1_Paramedic_Smith_Noisy_Results_NewVersion/"$command".txt
    elif [ $d == "5" ]
    then
        deepspeech --model deepspeech-0.9.3-models.pbmm --scorer deepspeech-0.9.3-models.scorer --audio $f > ./5_McLaren_EMT_Radio_Call_Alpha_107_Recording_Noisy_Results_NewVersion/"$command".txt
    elif [ $d == "6" ]
    then
        deepspeech --model deepspeech-0.9.3-models.pbmm --scorer deepspeech-0.9.3-models.scorer --audio $f > ./6_McLaren_EMT_Radio_Call_Alpha_117_Recording_Noisy_Results_NewVersion/"$command".txt
    else
        deepspeech --model deepspeech-0.9.3-models.pbmm --scorer deepspeech-0.9.3-models.scorer --audio $f > ./7_McLaren_EMT_Radio_Call_Alpha_101_Recording_Noisy_Results_NewVersion/"$command".txt
    fi
done
