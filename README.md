# REFERENCES

This code was initially used in 2019 in order to compare DeepSpeech v0.6 to the Google Speech Recognition tool.

The original code can be found at: https://github.com/Terry0923/Speech-Recognition-Assessment

# SPEECH RECOGNITION ASSESSMENT

This is the assessment for Mozilla DeepSpeech v0.9.3

There will be mutliple tests done in order to create a proper assessment

The first of these will be to get an intial baseline of DeepSpeech Performance

Secondly, the tests will be conducted again after the base scorer has been trained given the EMS ontology that is available

## NOISE PROFILE

1. [Clean] Clean Profile (No Noise)
2. [People_Talking] People Talking in the background (Low -> Medium -> High)
3. [Cafeteria_Noise] Noise from a cafeteria (Low -> Medium -> High)
4. [Sirens_Noise] Noise from sirens (Low -> Medium -> High)

Four EMS speech sample used: Paramedic Smith, EMT 107, EMT 101 (under the folder "audio")

# SOFTWARE NEEDED

The models for DeepSpeech are too large to push to github

Once the repository is cloned, open a terminal and run the following commands:

    wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm

    wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer

The scorer that was trained using the medical terms we possess is in the repository and will appear if cloned
