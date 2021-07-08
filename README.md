# SPEECH RECOGNITION ASSESSMENT

This is the assessment for Google Speech Recognition vs Deepspeech Recognition

# FOLDERS EXPLANATION

WER_generator_noise.py: code that generates the assessments

API_Results: google speech recognition text results
Results:  Deepspeech recongition text results

audio: audio files used

Golden_Transcript: correct transcription for audio files

No_Noise_Results: google speech recognition performance without noisy background

## NOISE PROFILE

1. [Clean] Clean Profile (No Noise)
2. [People_Talking] People Talking in the background (Low -> Medium -> High)
3. [Cafeteria_Noise] Noise from a cafeteria (Low -> Medium -> High)
4. [Sirens_Noise] Noise from sirens (Low -> Medium -> High)

Four EMS speech sample used: Paramedic Smith, EMT 107, EMT 101 (under the folder "audio")

## GRAPH EXPLANATION

### performance under different noise profiles

Clean, Low Noise, Medium Noise, High Noise denote the strength of the noise in the background

(Google) or (Mozilla) denote the speech recognition technique used

For example:

Clean (Google): Google speech recognition performance without noisy background

Clean (Mozilla): Mozilla Deepspeech Recognition performance without noisy background

#### cafe.png

This graph shows the speech recognition performance for both google speech recognition and deepspeech recogition assessed in a cafeteria.

#### ppl.png

This graph shows the speech recognition performance for both google speech recognition and deepspeech recogition assessed when people are talking in the background

#### sirens.png

This graph shows the speech recognition performance for both google speech recognition and deepspeech recogition assessed with sirens noise

### Overall Performance assessment

#### deep_ave.png

This graph shows the performance of DEEPSPEECH under different noise profiles

#### compare.png

This graph shows the comparison of performances between Google Speech Recognition and Deepspeech Recognition
