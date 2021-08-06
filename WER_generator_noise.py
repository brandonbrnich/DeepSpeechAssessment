import numpy as np
import re
import codecs
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
import pandas as pd

clean_deep_accuracy_arr = []
cafe_low_deep_accuracy_arr = []
cafe_medium_deep_accuracy_arr = []
cafe_high_deep_accuracy_arr = []
people_low_deep_accuracy_arr = []
people_medium_deep_accuracy_arr = []
people_high_deep_accuracy_arr = []
sirens_low_deep_accuracy_arr = []
sirens_medium_deep_accuracy_arr = []
sirens_high_deep_accuracy_arr = []
stop_words = set(stopwords.words('english'))

# convert numbers to English words for assessment purposes


def num2words(num):
    nums_20_90 = ['twenty', 'thirty', 'forty', 'fifty',
                  'sixty', 'seventy', 'eighty', 'ninety']
    nums_0_19 = ['zero', 'one', 'Two', 'three', 'four', 'five', 'six', 'seven', 'eight', "nine", 'ten',
                 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']
    nums_dict = {100: 'hundred', 1000: 'thousand',
                 1000000: 'million', 1000000000: 'billion'}
    if num < 20:
        return nums_0_19[num]
    if num < 100:
        return nums_20_90[int(num/10)-2] + ('' if num % 10 == 0 else ' ' + nums_0_19[num % 10])
    # find the largest key smaller than num
    maxkey = max([key for key in nums_dict.keys() if key <= num])
    return num2words(int(num/maxkey)) + ' ' + nums_dict[int(maxkey)] + ('' if num % maxkey == 0 else ' ' + num2words(num % maxkey))

# check if the word are numbers


def customizedIsNumerical(str):
    if str.isnumeric():
        return True
    else:
        # sometimes the program miss something like 96. or 96%, it should still be considered numbers
        str = str[:-1]
        if(str.isnumeric()):
            return True
        return False

# convert string containing numbers to string containing english words


def convertScript(str):
    newstr = ""
    for word in str.split():
        if customizedIsNumerical(word):
            if not word.isnumeric():
                word = word[:-1]
            word = num2words(int(word))
        newstr += word + " "
    return newstr

# class that compare text to find WER and ACC


class TextComp(object):
    def __init__(self, original_text, recognition_text, encoding='utf-8'):
        # original_path: path of the original text
        # recognition_path: path of the recognized text
        # encoding: specifies the encoding which is to be used for the file
        self.original_text = original_text
        self.recognition_text = recognition_text
        self.encoding = encoding
        self.I = 0
        self.S = 0
        self.D = 0

    def Preprocess(self, text):
        tokenizer = RegexpTokenizer(r'\w+')
        words = tokenizer.tokenize(text.lower())
        filtered_words = list(filter(lambda w: w not in stop_words, words))
        return filtered_words

    def WER(self, debug=False):
        r = self.Preprocess(self.original_text)
        h = self.Preprocess(self.recognition_text)
        # costs will holds the costs, like in the Levenshtein distance algorithm
        costs = [[0 for inner in range(len(h) + 1)]
                 for outer in range(len(r) + 1)]
        # backtrace will hold the operations we've done.
        # so we could later backtrace, like the WER algorithm requires us to.
        backtrace = [[0 for inner in range(len(h) + 1)]
                     for outer in range(len(r) + 1)]

        OP_OK = 0
        OP_SUB = 1
        OP_INS = 2
        OP_DEL = 3

        # First column represents the case where we achieve zero
        # hypothesis words by deleting all reference words.
        for i in range(1, len(r) + 1):
            costs[i][0] = i
            backtrace[i][0] = OP_DEL

        # First row represents the case where we achieve the hypothesis
        # by inserting all hypothesis words into a zero-length reference.
        for j in range(1, len(h) + 1):
            costs[0][j] = j
            backtrace[0][j] = OP_INS

        # computation
        for i in range(1, len(r) + 1):
            for j in range(1, len(h) + 1):
                if r[i - 1] == h[j - 1]:
                    costs[i][j] = costs[i - 1][j - 1]
                    backtrace[i][j] = OP_OK
                else:
                    # penalty is always 1
                    substitutionCost = costs[i - 1][j - 1] + 1
                    insertionCost = costs[i][j - 1] + 1  # penalty is always 1
                    deletionCost = costs[i - 1][j] + 1  # penalty is always 1

                    costs[i][j] = min(substitutionCost,
                                      insertionCost, deletionCost)
                    if costs[i][j] == substitutionCost:
                        backtrace[i][j] = OP_SUB
                    elif costs[i][j] == insertionCost:
                        backtrace[i][j] = OP_INS
                    else:
                        backtrace[i][j] = OP_DEL

        # back trace though the best route:
        i = len(r)
        j = len(h)
        self.S = 0
        self.D = 0
        self.I = 0
        numCor = 0
        if debug:
            print("OP\toriginal\trecognition")
            lines = []
        while i > 0 or j > 0:
            if backtrace[i][j] == OP_OK:
                numCor += 1
                i -= 1
                j -= 1
                if debug:
                    lines.append("OK\t" + r[i] + "\t" + h[j])
            elif backtrace[i][j] == OP_SUB:
                self.S += 1
                i -= 1
                j -= 1
                if debug:
                    lines.append("SUB\t" + r[i] + "\t" + h[j])
                    sub_arr.append("SUB\t" + r[i] + "\t" + h[j])
            elif backtrace[i][j] == OP_INS:
                self.I += 1
                # self.Insertions.append(OP_INS)
                j -= 1
                if debug:
                    lines.append("INS\t" + "****" + "\t" + h[j])
                    ins_arr.append("INS\t" + "****" + "\t" + h[j])
            elif backtrace[i][j] == OP_DEL:
                self.D += 1
                # self.Deletions.append(OP_DEL)
                i -= 1
                if debug:
                    lines.append("DEL\t" + r[i] + "\t" + "****")
                    del_arr.append("DEL\t" + r[i] + "\t" + "****")
        if debug:
            lines = reversed(lines)
            for line in lines:
                print(line)
            print("#cor " + str(numCor))
            print("#sub " + str(self.S))
            print("#del " + str(self.D))
            print("#ins " + str(self.I))
            return (self.S + self.D + self.I) / float(len(r))
        wer_result = round((self.S + self.D + self.I) / float(len(r)), 3)
        return wer_result

    def Accuracy(self):
        return float(len(self.Preprocess(self.original_text)) - self.D - self.S) / len(
            self.Preprocess(self.original_text))

    def getInsertions(self):
        return self.Insertions

    def getSubstitutions(self):
        return self.Substitutions

    def getDeletions(self):
        return self.Deletions


if __name__ == '__main__':

    debug = False

    # File 1

    # golden script
    with open("Golden_Transcript/1_Paramedic_Smith_Original_Transcript.txt", 'r') as myfile:
        original_first = myfile.read().replace('\n', '')
        original_first_deep = convertScript(original_first)

    # These files are from the 2019 evalution
    # clean
    with open("No_Noise_Results/1_Paramedic_Smith_Original_deep_2019.txt", 'r') as myfile:
        latest_deep_first = myfile.read().replace('\n', '')
    # cafe-low
    with open("1_Paramedic_Smith_Noisy_Results/1_Paramedic_Smith_Cafeteria_Low_Noise.txt", 'r') as myfile:
        latest_deep_first_cafe_low = myfile.read().replace('\n', '')
    # cafe-medium
    with open("1_Paramedic_Smith_Noisy_Results/1_Paramedic_Smith_Cafeteria_Medium_Noise.txt", 'r') as myfile:
        latest_deep_first_cafe_medium = myfile.read().replace('\n', '')
    # cafe-high
    with open("1_Paramedic_Smith_Noisy_Results/1_Paramedic_Smith_Cafeteria_High_Noise.txt", 'r') as myfile:
        latest_deep_first_cafe_high = myfile.read().replace('\n', '')
    # people-low
    with open("1_Paramedic_Smith_Noisy_Results/1_Paramedic_Smith_People_Low_Noise.txt", 'r') as myfile:
        latest_deep_first_people_low = myfile.read().replace('\n', '')
    # people-medium
    with open("1_Paramedic_Smith_Noisy_Results/1_Paramedic_Smith_People_Medium_Noise.txt", 'r') as myfile:
        latest_deep_first_people_medium = myfile.read().replace('\n', '')
    # people-high
    with open("1_Paramedic_Smith_Noisy_Results/1_Paramedic_Smith_People_High_Noise.txt", 'r') as myfile:
        latest_deep_first_people_high = myfile.read().replace('\n', '')
    # sirens-low
    with open("1_Paramedic_Smith_Noisy_Results/1_Paramedic_Smith_Sirens_Low_Noise.txt", 'r') as myfile:
        latest_deep_first_sirens_low = myfile.read().replace('\n', '')
    # sirens-medium
    with open("1_Paramedic_Smith_Noisy_Results/1_Paramedic_Smith_Sirens_Medium_Noise.txt", 'r') as myfile:
        latest_deep_first_sirens_medium = myfile.read().replace('\n', '')
    # sirens-high
    with open("1_Paramedic_Smith_Noisy_Results/1_Paramedic_Smith_Sirens_High_Noise.txt", 'r') as myfile:
        latest_deep_first_sirens_high = myfile.read().replace('\n', '')

    # evaluating the files from new version
    # clean
    with open("No_Noise_Results_Updated/1_Paramedic_Smith_Original.txt", 'r') as myfile:
        latest_deep_first_new = myfile.read().replace('\n', '')
    # cafe-low
    with open("1_Paramedic_Smith_Noisy_Results_NewVersion/1_Paramedic_Smith_Cafeteria_Low_Noise.txt", 'r') as myfile:
        latest_deep_first_cafe_low_new = myfile.read().replace('\n', '')
    # cafe-medium
    with open("1_Paramedic_Smith_Noisy_Results_NewVersion/1_Paramedic_Smith_Cafeteria_Medium_Noise.txt", 'r') as myfile:
        latest_deep_first_cafe_medium_new = myfile.read().replace('\n', '')
    # cafe-high
    with open("1_Paramedic_Smith_Noisy_Results_NewVersion/1_Paramedic_Smith_Cafeteria_High_Noise.txt", 'r') as myfile:
        latest_deep_first_cafe_high_new = myfile.read().replace('\n', '')
    # people-low
    with open("1_Paramedic_Smith_Noisy_Results_NewVersion/1_Paramedic_Smith_People_Low_Noise.txt", 'r') as myfile:
        latest_deep_first_people_low_new = myfile.read().replace('\n', '')
    # people-medium
    with open("1_Paramedic_Smith_Noisy_Results_NewVersion/1_Paramedic_Smith_People_Medium_Noise.txt", 'r') as myfile:
        latest_deep_first_people_medium_new = myfile.read().replace('\n', '')
    # people-high
    with open("1_Paramedic_Smith_Noisy_Results_NewVersion/1_Paramedic_Smith_People_High_Noise.txt", 'r') as myfile:
        latest_deep_first_people_high_new = myfile.read().replace('\n', '')
    # sirens-low
    with open("1_Paramedic_Smith_Noisy_Results_NewVersion/1_Paramedic_Smith_Sirens_Low_Noise.txt", 'r') as myfile:
        latest_deep_first_sirens_low_new = myfile.read().replace('\n', '')
    # sirens-medium
    with open("1_Paramedic_Smith_Noisy_Results_NewVersion/1_Paramedic_Smith_Sirens_Medium_Noise.txt", 'r') as myfile:
        latest_deep_first_sirens_medium_new = myfile.read().replace('\n', '')
    # sirens-high
    with open("1_Paramedic_Smith_Noisy_Results_NewVersion/1_Paramedic_Smith_Sirens_High_Noise.txt", 'r') as myfile:
        latest_deep_first_sirens_high_new = myfile.read().replace('\n', '')

    # audio files transcibed using the new model but with trained scorer
    # clean
    with open("No_Noise_Results_Improved/1_Paramedic_Smith_Original.txt", 'r') as myfile:
        latest_deep_first_improved = myfile.read().replace('\n', '')
    # cafe-low
    with open("1_Paramedic_Smith_Noisy_Results_NewVersion_Improved/1_Paramedic_Smith_Cafeteria_Low_Noise.txt", 'r') as myfile:
        latest_deep_first_cafe_low_improved = myfile.read().replace('\n', '')
    # cafe-medium
    with open("1_Paramedic_Smith_Noisy_Results_NewVersion_Improved/1_Paramedic_Smith_Cafeteria_Medium_Noise.txt", 'r') as myfile:
        latest_deep_first_cafe_medium_improved = myfile.read().replace('\n', '')
    # cafe-high
    with open("1_Paramedic_Smith_Noisy_Results_NewVersion_Improved/1_Paramedic_Smith_Cafeteria_High_Noise.txt", 'r') as myfile:
        latest_deep_first_cafe_high_improved = myfile.read().replace('\n', '')
    # people-low
    with open("1_Paramedic_Smith_Noisy_Results_NewVersion_Improved/1_Paramedic_Smith_People_Low_Noise.txt", 'r') as myfile:
        latest_deep_first_people_low_improved = myfile.read().replace('\n', '')
    # people-medium
    with open("1_Paramedic_Smith_Noisy_Results_NewVersion_Improved/1_Paramedic_Smith_People_Medium_Noise.txt", 'r') as myfile:
        latest_deep_first_people_medium_improved = myfile.read().replace('\n', '')
    # people-high
    with open("1_Paramedic_Smith_Noisy_Results_NewVersion_Improved/1_Paramedic_Smith_People_High_Noise.txt", 'r') as myfile:
        latest_deep_first_people_high_improved = myfile.read().replace('\n', '')
    # sirens-low
    with open("1_Paramedic_Smith_Noisy_Results_NewVersion_Improved/1_Paramedic_Smith_Sirens_Low_Noise.txt", 'r') as myfile:
        latest_deep_first_sirens_low_improved = myfile.read().replace('\n', '')
    # sirens-medium
    with open("1_Paramedic_Smith_Noisy_Results_NewVersion_Improved/1_Paramedic_Smith_Sirens_Medium_Noise.txt", 'r') as myfile:
        latest_deep_first_sirens_medium_improved = myfile.read().replace('\n', '')
    # sirens-high
    with open("1_Paramedic_Smith_Noisy_Results_NewVersion_Improved/1_Paramedic_Smith_Sirens_High_Noise.txt", 'r') as myfile:
        latest_deep_first_sirens_high_improved = myfile.read().replace('\n', '')

        ###Deepspeech Recognition###
    ###File 2###

    # golden script
    with open("Golden_Transcript/5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Original_Transcript.txt", 'r') as myfile:
        original_fifth = myfile.read().replace('\n', '')
        original_fifth_deep = convertScript(original_fifth)

    # old version data
    # clean
    with open("No_Noise_Results/5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Original_deep_2019.txt", 'r') as myfile:
        latest_deep_fifth = myfile.read().replace('\n', '')
        # cafe-low
    with open("5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Noisy_Results/5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Cafeteria_Low_Noise.txt", 'r') as myfile:
        latest_deep_fifth_cafe_low = myfile.read().replace('\n', '')
        # people-medium
    with open("5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Noisy_Results/5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_People_Medium_Noise.txt", 'r') as myfile:
        latest_deep_fifth_people_medium = myfile.read().replace('\n', '')
        # sirens-high
    with open("5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Noisy_Results/5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Sirens_High_Noise.txt", 'r') as myfile:
        latest_deep_fifth_sirens_high = myfile.read().replace('\n', '')
        # cafe-medium
    with open("5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Noisy_Results/5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Cafeteria_Medium_Noise.txt", 'r') as myfile:
        latest_deep_fifth_cafe_medium = myfile.read().replace('\n', '')
        # people-high
    with open("5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Noisy_Results/5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_People_High_Noise.txt", 'r') as myfile:
        latest_deep_fifth_people_high = myfile.read().replace('\n', '')
        # sirens-low
    with open("5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Noisy_Results/5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Sirens_Low_Noise.txt", 'r') as myfile:
        latest_deep_fifth_sirens_low = myfile.read().replace('\n', '')
        # cafe-high
    with open("5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Noisy_Results/5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Cafeteria_High_Noise.txt", 'r') as myfile:
        latest_deep_fifth_cafe_high = myfile.read().replace('\n', '')
        # people-low
    with open("5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Noisy_Results/5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_People_Low_Noise.txt", 'r') as myfile:
        latest_deep_fifth_people_low = myfile.read().replace('\n', '')
        # sirens-medium
    with open("5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Noisy_Results/5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Sirens_Medium_Noise.txt", 'r') as myfile:
        latest_deep_fifth_sirens_medium = myfile.read().replace('\n', '')

    # new version data
    # clean
    with open("No_Noise_Results_Updated/5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Original.txt", 'r') as myfile:
        latest_deep_fifth_new = myfile.read().replace('\n', '')
        # cafe-low
    with open("5_McLaren_EMT_Radio_Call_Alpha_107_Recording_Noisy_Results_NewVersion/5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Cafeteria_Low_Noise.txt", 'r') as myfile:
        latest_deep_fifth_cafe_low_new = myfile.read().replace('\n', '')
        # people-medium
    with open("5_McLaren_EMT_Radio_Call_Alpha_107_Recording_Noisy_Results_NewVersion/5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_People_Medium_Noise.txt", 'r') as myfile:
        latest_deep_fifth_people_medium_new = myfile.read().replace('\n', '')
        # sirens-high
    with open("5_McLaren_EMT_Radio_Call_Alpha_107_Recording_Noisy_Results_NewVersion/5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Sirens_High_Noise.txt", 'r') as myfile:
        latest_deep_fifth_sirens_high_new = myfile.read().replace('\n', '')
        # cafe-medium
    with open("5_McLaren_EMT_Radio_Call_Alpha_107_Recording_Noisy_Results_NewVersion/5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Cafeteria_Medium_Noise.txt", 'r') as myfile:
        latest_deep_fifth_cafe_medium_new = myfile.read().replace('\n', '')
        # people-high
    with open("5_McLaren_EMT_Radio_Call_Alpha_107_Recording_Noisy_Results_NewVersion/5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_People_High_Noise.txt", 'r') as myfile:
        latest_deep_fifth_people_high_new = myfile.read().replace('\n', '')
        # sirens-low
    with open("5_McLaren_EMT_Radio_Call_Alpha_107_Recording_Noisy_Results_NewVersion/5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Sirens_Low_Noise.txt", 'r') as myfile:
        latest_deep_fifth_sirens_low_new = myfile.read().replace('\n', '')
        # cafe-high
    with open("5_McLaren_EMT_Radio_Call_Alpha_107_Recording_Noisy_Results_NewVersion/5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Cafeteria_High_Noise.txt", 'r') as myfile:
        latest_deep_fifth_cafe_high_new = myfile.read().replace('\n', '')
        # people-low
    with open("5_McLaren_EMT_Radio_Call_Alpha_107_Recording_Noisy_Results_NewVersion/5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_People_Low_Noise.txt", 'r') as myfile:
        latest_deep_fifth_people_low_new = myfile.read().replace('\n', '')
        # sirens-medium
    with open("5_McLaren_EMT_Radio_Call_Alpha_107_Recording_Noisy_Results_NewVersion/5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Sirens_Medium_Noise.txt", 'r') as myfile:
        latest_deep_fifth_sirens_medium_new = myfile.read().replace('\n', '')

    # new version model with trained scorer
    # clean
    with open("No_Noise_Results_Improved/5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Original.txt", 'r') as myfile:
        latest_deep_fifth_improved = myfile.read().replace('\n', '')
        # cafe-low
    with open("5_McLaren_EMT_Radio_Call_Alpha_107_Recording_Noisy_Results_NewVersion_Improved/5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Cafeteria_Low_Noise.txt", 'r') as myfile:
        latest_deep_fifth_cafe_low_improved = myfile.read().replace('\n', '')
        # people-medium
    with open("5_McLaren_EMT_Radio_Call_Alpha_107_Recording_Noisy_Results_NewVersion_Improved/5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_People_Medium_Noise.txt", 'r') as myfile:
        latest_deep_fifth_people_medium_improved = myfile.read().replace('\n', '')
        # sirens-high
    with open("5_McLaren_EMT_Radio_Call_Alpha_107_Recording_Noisy_Results_NewVersion_Improved/5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Sirens_High_Noise.txt", 'r') as myfile:
        latest_deep_fifth_sirens_high_improved = myfile.read().replace('\n', '')
        # cafe-medium
    with open("5_McLaren_EMT_Radio_Call_Alpha_107_Recording_Noisy_Results_NewVersion_Improved/5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Cafeteria_Medium_Noise.txt", 'r') as myfile:
        latest_deep_fifth_cafe_medium_improved = myfile.read().replace('\n', '')
        # people-high
    with open("5_McLaren_EMT_Radio_Call_Alpha_107_Recording_Noisy_Results_NewVersion_Improved/5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_People_High_Noise.txt", 'r') as myfile:
        latest_deep_fifth_people_high_improved = myfile.read().replace('\n', '')
        # sirens-low
    with open("5_McLaren_EMT_Radio_Call_Alpha_107_Recording_Noisy_Results_NewVersion_Improved/5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Sirens_Low_Noise.txt", 'r') as myfile:
        latest_deep_fifth_sirens_low_improved = myfile.read().replace('\n', '')
        # cafe-high
    with open("5_McLaren_EMT_Radio_Call_Alpha_107_Recording_Noisy_Results_NewVersion_Improved/5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Cafeteria_High_Noise.txt", 'r') as myfile:
        latest_deep_fifth_cafe_high_improved = myfile.read().replace('\n', '')
        # people-low
    with open("5_McLaren_EMT_Radio_Call_Alpha_107_Recording_Noisy_Results_NewVersion_Improved/5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_People_Low_Noise.txt", 'r') as myfile:
        latest_deep_fifth_people_low_improved = myfile.read().replace('\n', '')
        # sirens-medium
    with open("5_McLaren_EMT_Radio_Call_Alpha_107_Recording_Noisy_Results_NewVersion_Improved/5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Sirens_Medium_Noise.txt", 'r') as myfile:
        latest_deep_fifth_sirens_medium_improved = myfile.read().replace('\n', '')

        ###Deepspeech Recognition###
    ###File 3###

    # golden script
    with open("Golden_Transcript/6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Original_Transcript.txt", 'r') as myfile:
        original_sixth = myfile.read().replace('\n', '')
        original_sixth_deep = convertScript(original_sixth)

    # old version
    # clean
    with open("No_Noise_Results/6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Original_deep_2019.txt", 'r') as myfile:
        latest_deep_sixth = myfile.read().replace('\n', '')
        # cafe-low
    with open("6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Noisy_Results/6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Cafeteria_Low_Noise.txt", 'r') as myfile:
        latest_deep_sixth_cafe_low = myfile.read().replace('\n', '')
        # cafe-medium
    with open("6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Noisy_Results/6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Cafeteria_Medium_Noise.txt", 'r') as myfile:
        latest_deep_sixth_cafe_medium = myfile.read().replace('\n', '')
        # cafe-high
    with open("6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Noisy_Results/6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Cafeteria_High_Noise.txt", 'r') as myfile:
        latest_deep_sixth_cafe_high = myfile.read().replace('\n', '')
        # people-low
    with open("6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Noisy_Results/6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_People_Low_Noise.txt", 'r') as myfile:
        latest_deep_sixth_people_low = myfile.read().replace('\n', '')
        # people-medium
    with open("6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Noisy_Results/6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_People_Medium_Noise.txt", 'r') as myfile:
        latest_deep_sixth_people_medium = myfile.read().replace('\n', '')
        # people-high
    with open("6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Noisy_Results/6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_People_High_Noise.txt", 'r') as myfile:
        latest_deep_sixth_people_high = myfile.read().replace('\n', '')
        # sirens-low
    with open("6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Noisy_Results/6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Sirens_Low_Noise.txt", 'r') as myfile:
        latest_deep_sixth_sirens_low = myfile.read().replace('\n', '')
        # sirens-medium
    with open("6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Noisy_Results/6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Sirens_Medium_Noise.txt", 'r') as myfile:
        latest_deep_sixth_sirens_medium = myfile.read().replace('\n', '')
        # sirens-high
    with open("6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Noisy_Results/6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Sirens_High_Noise.txt", 'r') as myfile:
        latest_deep_sixth_sirens_high = myfile.read().replace('\n', '')

    # new version
    # clean
    with open("No_Noise_Results_Updated/6_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Original.txt", 'r') as myfile:
        latest_deep_sixth_new = myfile.read().replace('\n', '')
        # cafe-low
    with open("6_McLaren_EMT_Radio_Call_Alpha_117_Recording_Noisy_Results_NewVersion/6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Cafeteria_Low_Noise.txt", 'r') as myfile:
        latest_deep_sixth_cafe_low_new = myfile.read().replace('\n', '')
        # cafe-medium
    with open("6_McLaren_EMT_Radio_Call_Alpha_117_Recording_Noisy_Results_NewVersion/6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Cafeteria_Medium_Noise.txt", 'r') as myfile:
        latest_deep_sixth_cafe_medium_new = myfile.read().replace('\n', '')
        # cafe-high
    with open("6_McLaren_EMT_Radio_Call_Alpha_117_Recording_Noisy_Results_NewVersion/6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Cafeteria_High_Noise.txt", 'r') as myfile:
        latest_deep_sixth_cafe_high_new = myfile.read().replace('\n', '')
        # people-low
    with open("6_McLaren_EMT_Radio_Call_Alpha_117_Recording_Noisy_Results_NewVersion/6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_People_Low_Noise.txt", 'r') as myfile:
        latest_deep_sixth_people_low_new = myfile.read().replace('\n', '')
        # people-medium
    with open("6_McLaren_EMT_Radio_Call_Alpha_117_Recording_Noisy_Results_NewVersion/6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_People_Medium_Noise.txt", 'r') as myfile:
        latest_deep_sixth_people_medium_new = myfile.read().replace('\n', '')
        # people-high
    with open("6_McLaren_EMT_Radio_Call_Alpha_117_Recording_Noisy_Results_NewVersion/6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_People_High_Noise.txt", 'r') as myfile:
        latest_deep_sixth_people_high_new = myfile.read().replace('\n', '')
        # sirens-low
    with open("6_McLaren_EMT_Radio_Call_Alpha_117_Recording_Noisy_Results_NewVersion/6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Sirens_Low_Noise.txt", 'r') as myfile:
        latest_deep_sixth_sirens_low_new = myfile.read().replace('\n', '')
        # sirens-medium
    with open("6_McLaren_EMT_Radio_Call_Alpha_117_Recording_Noisy_Results_NewVersion/6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Sirens_Medium_Noise.txt", 'r') as myfile:
        latest_deep_sixth_sirens_medium_new = myfile.read().replace('\n', '')
        # sirens-high
    with open("6_McLaren_EMT_Radio_Call_Alpha_117_Recording_Noisy_Results_NewVersion/6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Sirens_High_Noise.txt", 'r') as myfile:
        latest_deep_sixth_sirens_high_new = myfile.read().replace('\n', '')

    # new version with improved scorer
    # clean
    with open("No_Noise_Results_Improved/6_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Original.txt", 'r') as myfile:
        latest_deep_sixth_improved = myfile.read().replace('\n', '')
        # cafe-low
    with open("6_McLaren_EMT_Radio_Call_Alpha_117_Recording_Noisy_Results_NewVersion_Improved/6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Cafeteria_Low_Noise.txt", 'r') as myfile:
        latest_deep_sixth_cafe_low_improved = myfile.read().replace('\n', '')
        # cafe-medium
    with open("6_McLaren_EMT_Radio_Call_Alpha_117_Recording_Noisy_Results_NewVersion_Improved/6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Cafeteria_Medium_Noise.txt", 'r') as myfile:
        latest_deep_sixth_cafe_medium_improved = myfile.read().replace('\n', '')
        # cafe-high
    with open("6_McLaren_EMT_Radio_Call_Alpha_117_Recording_Noisy_Results_NewVersion_Improved/6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Cafeteria_High_Noise.txt", 'r') as myfile:
        latest_deep_sixth_cafe_high_improved = myfile.read().replace('\n', '')
        # people-low
    with open("6_McLaren_EMT_Radio_Call_Alpha_117_Recording_Noisy_Results_NewVersion_Improved/6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_People_Low_Noise.txt", 'r') as myfile:
        latest_deep_sixth_people_low_improved = myfile.read().replace('\n', '')
        # people-medium
    with open("6_McLaren_EMT_Radio_Call_Alpha_117_Recording_Noisy_Results_NewVersion_Improved/6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_People_Medium_Noise.txt", 'r') as myfile:
        latest_deep_sixth_people_medium_improved = myfile.read().replace('\n', '')
        # people-high
    with open("6_McLaren_EMT_Radio_Call_Alpha_117_Recording_Noisy_Results_NewVersion_Improved/6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_People_High_Noise.txt", 'r') as myfile:
        latest_deep_sixth_people_high_improved = myfile.read().replace('\n', '')
        # sirens-low
    with open("6_McLaren_EMT_Radio_Call_Alpha_117_Recording_Noisy_Results_NewVersion_Improved/6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Sirens_Low_Noise.txt", 'r') as myfile:
        latest_deep_sixth_sirens_low_improved = myfile.read().replace('\n', '')
        # sirens-medium
    with open("6_McLaren_EMT_Radio_Call_Alpha_117_Recording_Noisy_Results_NewVersion_Improved/6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Sirens_Medium_Noise.txt", 'r') as myfile:
        latest_deep_sixth_sirens_medium_improved = myfile.read().replace('\n', '')
        # sirens-high
    with open("6_McLaren_EMT_Radio_Call_Alpha_117_Recording_Noisy_Results_NewVersion_Improved/6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Sirens_High_Noise.txt", 'r') as myfile:
        latest_deep_sixth_sirens_high_improved = myfile.read().replace('\n', '')

        ###Deepspeech Recongition###
    ###File 4###

    # golden script
    with open("Golden_Transcript/7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Original_Transcript.txt", 'r') as myfile:
        original_seventh = myfile.read().replace('\n', '')
        original_seventh_deep = convertScript(original_seventh)

    # old version
    # clean
    with open("No_Noise_Results/7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Original_deep_2019.txt", 'r') as myfile:
        latest_deep_seventh = myfile.read().replace('\n', '')
        # cafe-low
    with open("7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Noisy_Results/7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Cafeteria_Low_Noise.txt", 'r') as myfile:
        latest_deep_seventh_cafe_low = myfile.read().replace('\n', '')
        # cafe-medium
    with open("7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Noisy_Results/7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Cafeteria_Medium_Noise.txt", 'r') as myfile:
        latest_deep_seventh_cafe_medium = myfile.read().replace('\n', '')
        # cafe-high
    with open("7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Noisy_Results/7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Cafeteria_High_Noise.txt", 'r') as myfile:
        latest_deep_seventh_cafe_high = myfile.read().replace('\n', '')
        # people-low
    with open("7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Noisy_Results/7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_People_Low_Noise.txt", 'r') as myfile:
        latest_deep_seventh_people_low = myfile.read().replace('\n', '')
        # people-medium
    with open("7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Noisy_Results/7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_People_Medium_Noise.txt", 'r') as myfile:
        latest_deep_seventh_people_medium = myfile.read().replace('\n', '')
        # people-high
    with open("7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Noisy_Results/7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_People_High_Noise.txt", 'r') as myfile:
        latest_deep_seventh_people_high = myfile.read().replace('\n', '')
        # sirens-low
    with open("7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Noisy_Results/7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Sirens_Low_Noise.txt", 'r') as myfile:
        latest_deep_seventh_sirens_low = myfile.read().replace('\n', '')
        # sirens-medium
    with open("7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Noisy_Results/7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Sirens_Medium_Noise.txt", 'r') as myfile:
        latest_deep_seventh_sirens_medium = myfile.read().replace('\n', '')
        # sirens-high
    with open("7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Noisy_Results/7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Sirens_High_Noise.txt", 'r') as myfile:
        latest_deep_seventh_sirens_high = myfile.read().replace('\n', '')

    ###Deepspeech Recongition###
    # new version
    # clean
    with open("No_Noise_Results_Updated/7_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Original.txt", 'r') as myfile:
        latest_deep_seventh_new = myfile.read().replace('\n', '')
        # cafe-low
    with open("7_McLaren_EMT_Radio_Call_Alpha_101_Recording_Noisy_Results_NewVersion/7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Cafeteria_Low_Noise.txt", 'r') as myfile:
        latest_deep_seventh_cafe_low_new = myfile.read().replace('\n', '')
        # cafe-medium
    with open("7_McLaren_EMT_Radio_Call_Alpha_101_Recording_Noisy_Results_NewVersion/7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Cafeteria_Medium_Noise.txt", 'r') as myfile:
        latest_deep_seventh_cafe_medium_new = myfile.read().replace('\n', '')
        # cafe-high
    with open("7_McLaren_EMT_Radio_Call_Alpha_101_Recording_Noisy_Results_NewVersion/7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Cafeteria_High_Noise.txt", 'r') as myfile:
        latest_deep_seventh_cafe_high_new = myfile.read().replace('\n', '')
        # people-low
    with open("7_McLaren_EMT_Radio_Call_Alpha_101_Recording_Noisy_Results_NewVersion/7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_People_Low_Noise.txt", 'r') as myfile:
        latest_deep_seventh_people_low_new = myfile.read().replace('\n', '')
        # people-medium
    with open("7_McLaren_EMT_Radio_Call_Alpha_101_Recording_Noisy_Results_NewVersion/7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_People_Medium_Noise.txt", 'r') as myfile:
        latest_deep_seventh_people_medium_new = myfile.read().replace('\n', '')
        # people-high
    with open("7_McLaren_EMT_Radio_Call_Alpha_101_Recording_Noisy_Results_NewVersion/7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_People_High_Noise.txt", 'r') as myfile:
        latest_deep_seventh_people_high_new = myfile.read().replace('\n', '')
        # sirens-low
    with open("7_McLaren_EMT_Radio_Call_Alpha_101_Recording_Noisy_Results_NewVersion/7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Sirens_Low_Noise.txt", 'r') as myfile:
        latest_deep_seventh_sirens_low_new = myfile.read().replace('\n', '')
        # sirens-medium
    with open("7_McLaren_EMT_Radio_Call_Alpha_101_Recording_Noisy_Results_NewVersion/7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Sirens_Medium_Noise.txt", 'r') as myfile:
        latest_deep_seventh_sirens_medium_new = myfile.read().replace('\n', '')
        # sirens-high
    with open("7_McLaren_EMT_Radio_Call_Alpha_101_Recording_Noisy_Results_NewVersion/7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Sirens_High_Noise.txt", 'r') as myfile:
        latest_deep_seventh_sirens_high_new = myfile.read().replace('\n', '')

    # new version with improved scorer
    # clean
    with open("No_Noise_Results_Improved/7_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Original.txt", 'r') as myfile:
        latest_deep_seventh_improved = myfile.read().replace('\n', '')
        # cafe-low
    with open("7_McLaren_EMT_Radio_Call_Alpha_101_Recording_Noisy_Results_NewVersion_Improved/7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Cafeteria_Low_Noise.txt", 'r') as myfile:
        latest_deep_seventh_cafe_low_improved = myfile.read().replace('\n', '')
        # cafe-medium
    with open("7_McLaren_EMT_Radio_Call_Alpha_101_Recording_Noisy_Results_NewVersion_Improved/7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Cafeteria_Medium_Noise.txt", 'r') as myfile:
        latest_deep_seventh_cafe_medium_improved = myfile.read().replace('\n', '')
        # cafe-high
    with open("7_McLaren_EMT_Radio_Call_Alpha_101_Recording_Noisy_Results_NewVersion_Improved/7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Cafeteria_High_Noise.txt", 'r') as myfile:
        latest_deep_seventh_cafe_high_improved = myfile.read().replace('\n', '')
        # people-low
    with open("7_McLaren_EMT_Radio_Call_Alpha_101_Recording_Noisy_Results_NewVersion_Improved/7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_People_Low_Noise.txt", 'r') as myfile:
        latest_deep_seventh_people_low_improved = myfile.read().replace('\n', '')
        # people-medium
    with open("7_McLaren_EMT_Radio_Call_Alpha_101_Recording_Noisy_Results_NewVersion_Improved/7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_People_Medium_Noise.txt", 'r') as myfile:
        latest_deep_seventh_people_medium_improved = myfile.read().replace('\n', '')
        # people-high
    with open("7_McLaren_EMT_Radio_Call_Alpha_101_Recording_Noisy_Results_NewVersion_Improved/7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_People_High_Noise.txt", 'r') as myfile:
        latest_deep_seventh_people_high_improved = myfile.read().replace('\n', '')
        # sirens-low
    with open("7_McLaren_EMT_Radio_Call_Alpha_101_Recording_Noisy_Results_NewVersion_Improved/7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Sirens_Low_Noise.txt", 'r') as myfile:
        latest_deep_seventh_sirens_low_improved = myfile.read().replace('\n', '')
        # sirens-medium
    with open("7_McLaren_EMT_Radio_Call_Alpha_101_Recording_Noisy_Results_NewVersion_Improved/7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Sirens_Medium_Noise.txt", 'r') as myfile:
        latest_deep_seventh_sirens_medium_improved = myfile.read().replace('\n', '')
        # sirens-high
    with open("7_McLaren_EMT_Radio_Call_Alpha_101_Recording_Noisy_Results_NewVersion_Improved/7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Sirens_High_Noise.txt", 'r') as myfile:
        latest_deep_seventh_sirens_high_improved = myfile.read().replace('\n', '')

    ######################GENERATE WER###########################

    # first file
        ### Old Version of DeepSpeech ###
    # clean
    latest_deep_stats_first = TextComp(latest_deep_first, original_first_deep)
    print("[clean_deepspeech_first_2019] Word Error Rate:" +
          str(latest_deep_stats_first.WER(debug)))
    print("[clean_deepspeech_first_2019] Accuracy:" +
          str(latest_deep_stats_first.Accuracy()))
    clean_deep_accuracy_arr.append(latest_deep_stats_first.Accuracy())
    # cafe-low
    latest_deep_stats_first_cafe_low = TextComp(
        latest_deep_first_cafe_low, original_first_deep)
    print("[cafe_low_deepspeech_first_2019] Word Error Rate:" +
          str(latest_deep_stats_first_cafe_low.WER(debug)))
    print("[cafe_low_deepspeech_first_2019] Accuracy:" +
          str(latest_deep_stats_first_cafe_low.Accuracy()))
    cafe_low_deep_accuracy_arr.append(
        latest_deep_stats_first_cafe_low.Accuracy())
    # cafe-medium
    latest_deep_stats_first_cafe_medium = TextComp(
        latest_deep_first_cafe_medium, original_first_deep)
    print("[cafe_medium_deepspeech_first_2019] Word Error Rate:" +
          str(latest_deep_stats_first_cafe_medium.WER(debug)))
    print("[cafe_medium_deepspeech_first_2019] Accuracy:" +
          str(latest_deep_stats_first_cafe_medium.Accuracy()))
    cafe_medium_deep_accuracy_arr.append(
        latest_deep_stats_first_cafe_medium.Accuracy())
    # cafe-high
    latest_deep_stats_first_cafe_high = TextComp(
        latest_deep_first_cafe_high, original_first_deep)
    print("[cafe_high_deepspeech_first_2019] Word Error Rate:" +
          str(latest_deep_stats_first_cafe_high.WER(debug)))
    print("[cafe_high_deepspeech_first_2019] Accuracy:" +
          str(latest_deep_stats_first_cafe_high.Accuracy()))
    cafe_high_deep_accuracy_arr.append(
        latest_deep_stats_first_cafe_high.Accuracy())
    # people-low
    latest_deep_stats_first_people_low = TextComp(
        latest_deep_first_people_low, original_first_deep)
    print("[people_low_deepspeech_first_2019] Word Error Rate:" +
          str(latest_deep_stats_first_people_low.WER(debug)))
    print("[people_low_deepspeech_first_2019] Accuracy:" +
          str(latest_deep_stats_first_people_low.Accuracy()))
    people_low_deep_accuracy_arr.append(
        latest_deep_stats_first_people_low.Accuracy())
    # people-medium
    latest_deep_stats_first_people_medium = TextComp(
        latest_deep_first_people_medium, original_first_deep)
    print("_people_medium_deepspeech_first_2019] Word Error Rate:" +
          str(latest_deep_stats_first_people_medium.WER(debug)))
    print("[people_medium_deepspeech_first_2019] Accuracy:" +
          str(latest_deep_stats_first_people_medium.Accuracy()))
    people_medium_deep_accuracy_arr.append(
        latest_deep_stats_first_people_medium.Accuracy())
    # people-high
    latest_deep_stats_first_people_high = TextComp(
        latest_deep_first_people_high, original_first_deep)
    print("[people_high_deepspeech_first_2019] Word Error Rate:" +
          str(latest_deep_stats_first_people_high.WER(debug)))
    print("[people_high_deepspeech_first_2019] Accuracy:" +
          str(latest_deep_stats_first_people_high.Accuracy()))
    people_high_deep_accuracy_arr.append(
        latest_deep_stats_first_people_high.Accuracy())
    # sirens-low
    latest_deep_stats_first_sirens_low = TextComp(
        latest_deep_first_sirens_low, original_first_deep)
    print("[sirens_low_deepspeech_first_2019] Word Error Rate:" +
          str(latest_deep_stats_first_sirens_low.WER(debug)))
    print("[sirens_low_deepspeech_first_2019] Accuracy:" +
          str(latest_deep_stats_first_sirens_low.Accuracy()))
    sirens_low_deep_accuracy_arr.append(
        latest_deep_stats_first_sirens_low.Accuracy())
    # sirens-medium
    latest_deep_stats_first_sirens_medium = TextComp(
        latest_deep_first_sirens_medium, original_first_deep)
    print("[sirens_medium_deepspeech_first_2019] Word Error Rate:" +
          str(latest_deep_stats_first_sirens_medium.WER(debug)))
    print("[sirens_medium_deepspeech_first_2019] Accuracy:" +
          str(latest_deep_stats_first_sirens_medium.Accuracy()))
    sirens_medium_deep_accuracy_arr.append(
        latest_deep_stats_first_sirens_medium.Accuracy())
    # sirens-high
    latest_deep_stats_first_sirens_high = TextComp(
        latest_deep_first_sirens_high, original_first_deep)
    print("[sirens_high_deepspeech_first_2019] Word Error Rate:" +
          str(latest_deep_stats_first_sirens_high.WER(debug)))
    print("[sirens_high_deepspeech_first_2019] Accuracy:" +
          str(latest_deep_stats_first_sirens_high.Accuracy()))
    sirens_high_deep_accuracy_arr.append(
        latest_deep_stats_first_sirens_high.Accuracy())

    ### New version of DeepSpeech###
    # clean
    latest_deep_stats_first_new = TextComp(
        latest_deep_first_new, original_first_deep)
    print("[clean_deepspeech_first_2021] Word Error Rate:" +
          str(latest_deep_stats_first_new.WER(debug)))
    print("[clean_deepspeech_first_2021] Accuracy:" +
          str(latest_deep_stats_first_new.Accuracy()))
    clean_deep_accuracy_arr.append(latest_deep_stats_first_new.Accuracy())
    # cafe-low
    latest_deep_stats_first_cafe_low_new = TextComp(
        latest_deep_first_cafe_low_new, original_first_deep)
    print("[cafe_low_deepspeech_first_2021] Word Error Rate:" +
          str(latest_deep_stats_first_cafe_low_new.WER(debug)))
    print("[cafe_low_deepspeech_first_2021] Accuracy:" +
          str(latest_deep_stats_first_cafe_low_new.Accuracy()))
    cafe_low_deep_accuracy_arr.append(
        latest_deep_stats_first_cafe_low_new.Accuracy())
    # cafe-medium
    latest_deep_stats_first_cafe_medium_new = TextComp(
        latest_deep_first_cafe_medium_new, original_first_deep)
    print("[cafe_medium_deepspeech_first_2021] Word Error Rate:" +
          str(latest_deep_stats_first_cafe_medium_new.WER(debug)))
    print("[cafe_medium_deepspeech_first_2021] Accuracy:" +
          str(latest_deep_stats_first_cafe_medium_new.Accuracy()))
    cafe_medium_deep_accuracy_arr.append(
        latest_deep_stats_first_cafe_medium_new.Accuracy())
    # cafe-high
    latest_deep_stats_first_cafe_high_new = TextComp(
        latest_deep_first_cafe_high_new, original_first_deep)
    print("[cafe_high_deepspeech_first_2021] Word Error Rate:" +
          str(latest_deep_stats_first_cafe_high_new.WER(debug)))
    print("[cafe_high_deepspeech_first_2021] Accuracy:" +
          str(latest_deep_stats_first_cafe_high_new.Accuracy()))
    cafe_high_deep_accuracy_arr.append(
        latest_deep_stats_first_cafe_high_new.Accuracy())
    # people-low
    latest_deep_stats_first_people_low_new = TextComp(
        latest_deep_first_people_low_new, original_first_deep)
    print("[people_low_deepspeech_first_2021] Word Error Rate:" +
          str(latest_deep_stats_first_people_low_new.WER(debug)))
    print("[people_low_deepspeech_first_2021] Accuracy:" +
          str(latest_deep_stats_first_people_low_new.Accuracy()))
    people_low_deep_accuracy_arr.append(
        latest_deep_stats_first_people_low_new.Accuracy())
    # people-medium
    latest_deep_stats_first_people_medium_new = TextComp(
        latest_deep_first_people_medium_new, original_first_deep)
    print("_people_medium_deepspeech_first_2021] Word Error Rate:" +
          str(latest_deep_stats_first_people_medium_new.WER(debug)))
    print("[people_medium_deepspeech_first_2021] Accuracy:" +
          str(latest_deep_stats_first_people_medium_new.Accuracy()))
    people_medium_deep_accuracy_arr.append(
        latest_deep_stats_first_people_medium_new.Accuracy())
    # people-high
    latest_deep_stats_first_people_high_new = TextComp(
        latest_deep_first_people_high_new, original_first_deep)
    print("[people_high_deepspeech_first_2021] Word Error Rate:" +
          str(latest_deep_stats_first_people_high_new.WER(debug)))
    print("[people_high_deepspeech_first_2021] Accuracy:" +
          str(latest_deep_stats_first_people_high_new.Accuracy()))
    people_high_deep_accuracy_arr.append(
        latest_deep_stats_first_people_high_new.Accuracy())
    # sirens-low
    latest_deep_stats_first_sirens_low_new = TextComp(
        latest_deep_first_sirens_low_new, original_first_deep)
    print("[sirens_low_deepspeech_first_2021] Word Error Rate:" +
          str(latest_deep_stats_first_sirens_low_new.WER(debug)))
    print("[sirens_low_deepspeech_first_2021] Accuracy:" +
          str(latest_deep_stats_first_sirens_low_new.Accuracy()))
    sirens_low_deep_accuracy_arr.append(
        latest_deep_stats_first_sirens_low_new.Accuracy())
    # sirens-medium
    latest_deep_stats_first_sirens_medium_new = TextComp(
        latest_deep_first_sirens_medium_new, original_first_deep)
    print("[sirens_medium_deepspeech_first_2021] Word Error Rate:" +
          str(latest_deep_stats_first_sirens_medium_new.WER(debug)))
    print("[sirens_medium_deepspeech_first_2021] Accuracy:" +
          str(latest_deep_stats_first_sirens_medium_new.Accuracy()))
    sirens_medium_deep_accuracy_arr.append(
        latest_deep_stats_first_sirens_medium_new.Accuracy())
    # sirens-high
    latest_deep_stats_first_sirens_high_new = TextComp(
        latest_deep_first_sirens_high_new, original_first_deep)
    print("[sirens_high_deepspeech_first_2021] Word Error Rate:" +
          str(latest_deep_stats_first_sirens_high_new.WER(debug)))
    print("[sirens_high_deepspeech_first_2021] Accuracy:" +
          str(latest_deep_stats_first_sirens_high_new.Accuracy()))
    sirens_high_deep_accuracy_arr.append(
        latest_deep_stats_first_sirens_high_new.Accuracy())
    
     ### New version of DeepSpeech with improved scorer###
    # clean
    latest_deep_stats_first_improved = TextComp(
        latest_deep_first_improved, original_first_deep)
    print("[clean_deepspeech_first_2021_imp] Word Error Rate:" +
          str(latest_deep_stats_first_improved.WER(debug)))
    print("[clean_deepspeech_first_2021_imp] Accuracy:" +
          str(latest_deep_stats_first_improved.Accuracy()))
    clean_deep_accuracy_arr.append(latest_deep_stats_first_improved.Accuracy())
    # cafe-low
    latest_deep_stats_first_cafe_low_improved = TextComp(
        latest_deep_first_cafe_low_improved, original_first_deep)
    print("[cafe_low_deepspeech_first_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_first_cafe_low_improved.WER(debug)))
    print("[cafe_low_deepspeech_first_2021_improved] Accuracy:" +
          str(latest_deep_stats_first_cafe_low_improved.Accuracy()))
    cafe_low_deep_accuracy_arr.append(
        latest_deep_stats_first_cafe_low_improved.Accuracy())
    # cafe-medium
    latest_deep_stats_first_cafe_medium_improved = TextComp(
        latest_deep_first_cafe_medium_improved, original_first_deep)
    print("[cafe_medium_deepspeech_first_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_first_cafe_medium_improved.WER(debug)))
    print("[cafe_medium_deepspeech_first_2021_improved] Accuracy:" +
          str(latest_deep_stats_first_cafe_medium_improved.Accuracy()))
    cafe_medium_deep_accuracy_arr.append(
        latest_deep_stats_first_cafe_medium_improved.Accuracy())
    # cafe-high
    latest_deep_stats_first_cafe_high_improved = TextComp(
        latest_deep_first_cafe_high_improved, original_first_deep)
    print("[cafe_high_deepspeech_first_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_first_cafe_high_improved.WER(debug)))
    print("[cafe_high_deepspeech_first_2021_improved] Accuracy:" +
          str(latest_deep_stats_first_cafe_high_improved.Accuracy()))
    cafe_high_deep_accuracy_arr.append(
        latest_deep_stats_first_cafe_high_improved.Accuracy())
    # people-low
    latest_deep_stats_first_people_low_improved = TextComp(
        latest_deep_first_people_low_improved, original_first_deep)
    print("[people_low_deepspeech_first_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_first_people_low_improved.WER(debug)))
    print("[people_low_deepspeech_first_2021_improved] Accuracy:" +
          str(latest_deep_stats_first_people_low_improved.Accuracy()))
    people_low_deep_accuracy_arr.append(
        latest_deep_stats_first_people_low_improved.Accuracy())
    # people-medium
    latest_deep_stats_first_people_medium_improved = TextComp(
        latest_deep_first_people_medium_improved, original_first_deep)
    print("_people_medium_deepspeech_first_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_first_people_medium_improved.WER(debug)))
    print("[people_medium_deepspeech_first_2021_improved] Accuracy:" +
          str(latest_deep_stats_first_people_medium_improved.Accuracy()))
    people_medium_deep_accuracy_arr.append(
        latest_deep_stats_first_people_medium_improved.Accuracy())
    # people-high
    latest_deep_stats_first_people_high_improved = TextComp(
        latest_deep_first_people_high_improved, original_first_deep)
    print("[people_high_deepspeech_first_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_first_people_high_improved.WER(debug)))
    print("[people_high_deepspeech_first_2021_improved] Accuracy:" +
          str(latest_deep_stats_first_people_high_improved.Accuracy()))
    people_high_deep_accuracy_arr.append(
        latest_deep_stats_first_people_high_improved.Accuracy())
    # sirens-low
    latest_deep_stats_first_sirens_low_improved = TextComp(
        latest_deep_first_sirens_low_improved, original_first_deep)
    print("[sirens_low_deepspeech_first_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_first_sirens_low_improved.WER(debug)))
    print("[sirens_low_deepspeech_first_2021_improved] Accuracy:" +
          str(latest_deep_stats_first_sirens_low_improved.Accuracy()))
    sirens_low_deep_accuracy_arr.append(
        latest_deep_stats_first_sirens_low_improved.Accuracy())
    # sirens-medium
    latest_deep_stats_first_sirens_medium_improved = TextComp(
        latest_deep_first_sirens_medium_improved, original_first_deep)
    print("[sirens_medium_deepspeech_first_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_first_sirens_medium_improved.WER(debug)))
    print("[sirens_medium_deepspeech_first_2021_improved] Accuracy:" +
          str(latest_deep_stats_first_sirens_medium_improved.Accuracy()))
    sirens_medium_deep_accuracy_arr.append(
        latest_deep_stats_first_sirens_medium_improved.Accuracy())
    # sirens-high
    latest_deep_stats_first_sirens_high_improved = TextComp(
        latest_deep_first_sirens_high_improved, original_first_deep)
    print("[sirens_high_deepspeech_first_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_first_sirens_high_improved.WER(debug)))
    print("[sirens_high_deepspeech_first_2021_improved] Accuracy:" +
          str(latest_deep_stats_first_sirens_high_improved.Accuracy()))
    sirens_high_deep_accuracy_arr.append(
        latest_deep_stats_first_sirens_high_improved.Accuracy())

    # fifth file
    ### Old Version of DeepSpeech ###
    # clean
    latest_deep_stats_fifth = TextComp(latest_deep_fifth, original_fifth_deep)
    print("[clean_deepspeech_fifth_2019] Word Error Rate:" +
          str(latest_deep_stats_fifth.WER(debug)))
    print("[clean_deepspeech_fifth_2019] Accuracy:" +
          str(latest_deep_stats_fifth.Accuracy()))
    clean_deep_accuracy_arr.append(latest_deep_stats_fifth.Accuracy())
    # cafe-low
    latest_deep_stats_fifth_cafe_low = TextComp(
        latest_deep_fifth_cafe_low, original_fifth_deep)
    print("[cafe_low_deepspeech_fifth_2019] Word Error Rate:" +
          str(latest_deep_stats_fifth_cafe_low.WER(debug)))
    print("[cafe_low_deepspeech_fifth_2019] Accuracy:" +
          str(latest_deep_stats_fifth_cafe_low.Accuracy()))
    cafe_low_deep_accuracy_arr.append(
        latest_deep_stats_fifth_cafe_low.Accuracy())
    # cafe-medium
    latest_deep_stats_fifth_cafe_medium = TextComp(
        latest_deep_fifth_cafe_medium, original_fifth_deep)
    print("[cafe_medium_deepspeech_fifth_2019] Word Error Rate:" +
          str(latest_deep_stats_fifth_cafe_medium.WER(debug)))
    print("[cafe_medium_deepspeech_fifth_2019] Accuracy:" +
          str(latest_deep_stats_fifth_cafe_medium.Accuracy()))
    cafe_medium_deep_accuracy_arr.append(
        latest_deep_stats_fifth_cafe_medium.Accuracy())
    # cafe-high
    latest_deep_stats_fifth_cafe_high = TextComp(
        latest_deep_fifth_cafe_high, original_fifth_deep)
    print("[cafe_high_deepspeech_fifth_2019] Word Error Rate:" +
          str(latest_deep_stats_fifth_cafe_high.WER(debug)))
    print("[cafe_high_deepspeech_fifth_2019] Accuracy:" +
          str(latest_deep_stats_fifth_cafe_high.Accuracy()))
    cafe_high_deep_accuracy_arr.append(
        latest_deep_stats_fifth_cafe_high.Accuracy())
    # people-low
    latest_deep_stats_fifth_people_low = TextComp(
        latest_deep_fifth_people_low, original_fifth_deep)
    print("[people_low_deepspeech_fifth_2019] Word Error Rate:" +
          str(latest_deep_stats_fifth_people_low.WER(debug)))
    print("[people_low_deepspeech_fifth_2019] Accuracy:" +
          str(latest_deep_stats_fifth_people_low.Accuracy()))
    people_low_deep_accuracy_arr.append(
        latest_deep_stats_fifth_people_low.Accuracy())
    # people-medium
    latest_deep_stats_fifth_people_medium = TextComp(
        latest_deep_fifth_people_medium, original_fifth_deep)
    print("[people_medium_deepspeech_fifth_2019] Word Error Rate:" +
          str(latest_deep_stats_fifth_people_medium.WER(debug)))
    print("[people_medium_deepspeech_fifth_2019] Accuracy:" +
          str(latest_deep_stats_fifth_people_medium.Accuracy()))
    people_medium_deep_accuracy_arr.append(
        latest_deep_stats_fifth_people_medium.Accuracy())
    # people-high
    latest_deep_stats_fifth_people_high = TextComp(
        latest_deep_fifth_people_high, original_fifth_deep)
    print("[people_high_deepspeech_fifth_2019] Word Error Rate:" +
          str(latest_deep_stats_fifth_people_high.WER(debug)))
    print("[people_high_deepspeech_fifth_2019] Accuracy:" +
          str(latest_deep_stats_fifth_people_high.Accuracy()))
    people_high_deep_accuracy_arr.append(
        latest_deep_stats_fifth_people_high.Accuracy())
    # sirens-low
    latest_deep_stats_fifth_sirens_low = TextComp(
        latest_deep_fifth_sirens_low, original_fifth_deep)
    print("[sirens_low_deepspeech_fifth_2019] Word Error Rate:" +
          str(latest_deep_stats_fifth_sirens_low.WER(debug)))
    print("[sirens_low_deepspeech_fifth_2019] Accuracy:" +
          str(latest_deep_stats_fifth_sirens_low.Accuracy()))
    sirens_low_deep_accuracy_arr.append(
        latest_deep_stats_fifth_sirens_low.Accuracy())
    # sirens-medium
    latest_deep_stats_fifth_sirens_medium = TextComp(
        latest_deep_fifth_sirens_medium, original_fifth_deep)
    print("[sirens_medium_deepspeech_fifth_2019] Word Error Rate:" +
          str(latest_deep_stats_fifth_sirens_medium.WER(debug)))
    print("[sirens_medium_deepspeech_fifth_2019] Accuracy:" +
          str(latest_deep_stats_fifth_sirens_medium.Accuracy()))
    sirens_medium_deep_accuracy_arr.append(
        latest_deep_stats_fifth_sirens_medium.Accuracy())
    # sirens-high
    latest_deep_stats_fifth_sirens_high = TextComp(
        latest_deep_fifth_sirens_high, original_fifth_deep)
    print("[sirens_high_deepspeech_fifth_2019] Word Error Rate:" +
          str(latest_deep_stats_fifth_sirens_high.WER(debug)))
    print("[sirens_high_deepspeech_fifth_2019] Accuracy:" +
          str(latest_deep_stats_fifth_sirens_high.Accuracy()))
    sirens_high_deep_accuracy_arr.append(
        latest_deep_stats_fifth_sirens_high.Accuracy())

    ### New version of DeepSpeech###
    # clean
    latest_deep_stats_fifth_new = TextComp(
        latest_deep_fifth_new, original_fifth_deep)
    print("[clean_deepspeech_fifth_2021] Word Error Rate:" +
          str(latest_deep_stats_fifth_new.WER(debug)))
    print("[clean_deepspeech_fifth_2021] Accuracy:" +
          str(latest_deep_stats_fifth_new.Accuracy()))
    clean_deep_accuracy_arr.append(latest_deep_stats_fifth_new.Accuracy())
    # cafe-low
    latest_deep_stats_fifth_cafe_low_new = TextComp(
        latest_deep_fifth_cafe_low_new, original_fifth_deep)
    print("[cafe_low_deepspeech_fifth_2021] Word Error Rate:" +
          str(latest_deep_stats_fifth_cafe_low_new.WER(debug)))
    print("[cafe_low_deepspeech_fifth_2021] Accuracy:" +
          str(latest_deep_stats_fifth_cafe_low_new.Accuracy()))
    cafe_low_deep_accuracy_arr.append(
        latest_deep_stats_fifth_cafe_low_new.Accuracy())
    # cafe-medium
    latest_deep_stats_fifth_cafe_medium_new = TextComp(
        latest_deep_fifth_cafe_medium_new, original_fifth_deep)
    print("[cafe_medium_deepspeech_fifth_2021] Word Error Rate:" +
          str(latest_deep_stats_fifth_cafe_medium_new.WER(debug)))
    print("[cafe_medium_deepspeech_fifth_2021] Accuracy:" +
          str(latest_deep_stats_fifth_cafe_medium_new.Accuracy()))
    cafe_medium_deep_accuracy_arr.append(
        latest_deep_stats_fifth_cafe_medium_new.Accuracy())
    # cafe-high
    latest_deep_stats_fifth_cafe_high_new = TextComp(
        latest_deep_fifth_cafe_high_new, original_fifth_deep)
    print("[cafe_high_deepspeech_fifth_2021] Word Error Rate:" +
          str(latest_deep_stats_fifth_cafe_high_new.WER(debug)))
    print("[cafe_high_deepspeech_fifth_2021] Accuracy:" +
          str(latest_deep_stats_fifth_cafe_high_new.Accuracy()))
    cafe_high_deep_accuracy_arr.append(
        latest_deep_stats_fifth_cafe_high_new.Accuracy())
    # people-low
    latest_deep_stats_fifth_people_low_new = TextComp(
        latest_deep_fifth_people_low_new, original_fifth_deep)
    print("[people_low_deepspeech_fifth_2021] Word Error Rate:" +
          str(latest_deep_stats_fifth_people_low_new.WER(debug)))
    print("[people_low_deepspeech_fifth_2021] Accuracy:" +
          str(latest_deep_stats_fifth_people_low_new.Accuracy()))
    people_low_deep_accuracy_arr.append(
        latest_deep_stats_fifth_people_low_new.Accuracy())
    # people-medium
    latest_deep_stats_fifth_people_medium_new = TextComp(
        latest_deep_fifth_people_medium_new, original_fifth_deep)
    print("_people_medium_deepspeech_fifth_2021] Word Error Rate:" +
          str(latest_deep_stats_fifth_people_medium_new.WER(debug)))
    print("[people_medium_deepspeech_fifth_2021] Accuracy:" +
          str(latest_deep_stats_fifth_people_medium_new.Accuracy()))
    people_medium_deep_accuracy_arr.append(
        latest_deep_stats_fifth_people_medium_new.Accuracy())
    # people-high
    latest_deep_stats_fifth_people_high_new = TextComp(
        latest_deep_fifth_people_high_new, original_fifth_deep)
    print("[people_high_deepspeech_fifth_2021] Word Error Rate:" +
          str(latest_deep_stats_fifth_people_high_new.WER(debug)))
    print("[people_high_deepspeech_fifth_2021] Accuracy:" +
          str(latest_deep_stats_fifth_people_high_new.Accuracy()))
    people_high_deep_accuracy_arr.append(
        latest_deep_stats_fifth_people_high_new.Accuracy())
    # sirens-low
    latest_deep_stats_fifth_sirens_low_new = TextComp(
        latest_deep_fifth_sirens_low_new, original_fifth_deep)
    print("[sirens_low_deepspeech_fifth_2021] Word Error Rate:" +
          str(latest_deep_stats_fifth_sirens_low_new.WER(debug)))
    print("[sirens_low_deepspeech_fifth_2021] Accuracy:" +
          str(latest_deep_stats_fifth_sirens_low_new.Accuracy()))
    sirens_low_deep_accuracy_arr.append(
        latest_deep_stats_fifth_sirens_low_new.Accuracy())
    # sirens-medium
    latest_deep_stats_fifth_sirens_medium_new = TextComp(
        latest_deep_fifth_sirens_medium_new, original_fifth_deep)
    print("[sirens_medium_deepspeech_fifth_2021] Word Error Rate:" +
          str(latest_deep_stats_fifth_sirens_medium_new.WER(debug)))
    print("[sirens_medium_deepspeech_fifth_2021] Accuracy:" +
          str(latest_deep_stats_fifth_sirens_medium_new.Accuracy()))
    sirens_medium_deep_accuracy_arr.append(
        latest_deep_stats_fifth_sirens_medium_new.Accuracy())
    # sirens-high
    latest_deep_stats_fifth_sirens_high_new = TextComp(
        latest_deep_fifth_sirens_high_new, original_fifth_deep)
    print("[sirens_high_deepspeech_fifth_2021] Word Error Rate:" +
          str(latest_deep_stats_fifth_sirens_high_new.WER(debug)))
    print("[sirens_high_deepspeech_fifth_2021] Accuracy:" +
          str(latest_deep_stats_fifth_sirens_high_new.Accuracy()))
    sirens_high_deep_accuracy_arr.append(
        latest_deep_stats_fifth_sirens_high_new.Accuracy())
    
     ### New version of DeepSpeech with improved scorer###
    # clean
    latest_deep_stats_fifth_improved = TextComp(
        latest_deep_fifth_improved, original_fifth_deep)
    print("[clean_deepspeech_fifth_2021_imp] Word Error Rate:" +
          str(latest_deep_stats_fifth_improved.WER(debug)))
    print("[clean_deepspeech_fifth_2021_imp] Accuracy:" +
          str(latest_deep_stats_fifth_improved.Accuracy()))
    clean_deep_accuracy_arr.append(latest_deep_stats_fifth_improved.Accuracy())
    # cafe-low
    latest_deep_stats_fifth_cafe_low_improved = TextComp(
        latest_deep_fifth_cafe_low_improved, original_fifth_deep)
    print("[cafe_low_deepspeech_fifth_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_fifth_cafe_low_improved.WER(debug)))
    print("[cafe_low_deepspeech_fifth_2021_improved] Accuracy:" +
          str(latest_deep_stats_fifth_cafe_low_improved.Accuracy()))
    cafe_low_deep_accuracy_arr.append(
        latest_deep_stats_fifth_cafe_low_improved.Accuracy())
    # cafe-medium
    latest_deep_stats_fifth_cafe_medium_improved = TextComp(
        latest_deep_fifth_cafe_medium_improved, original_fifth_deep)
    print("[cafe_medium_deepspeech_fifth_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_fifth_cafe_medium_improved.WER(debug)))
    print("[cafe_medium_deepspeech_fifth_2021_improved] Accuracy:" +
          str(latest_deep_stats_fifth_cafe_medium_improved.Accuracy()))
    cafe_medium_deep_accuracy_arr.append(
        latest_deep_stats_fifth_cafe_medium_improved.Accuracy())
    # cafe-high
    latest_deep_stats_fifth_cafe_high_improved = TextComp(
        latest_deep_fifth_cafe_high_improved, original_fifth_deep)
    print("[cafe_high_deepspeech_fifth_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_fifth_cafe_high_improved.WER(debug)))
    print("[cafe_high_deepspeech_fifth_2021_improved] Accuracy:" +
          str(latest_deep_stats_fifth_cafe_high_improved.Accuracy()))
    cafe_high_deep_accuracy_arr.append(
        latest_deep_stats_fifth_cafe_high_improved.Accuracy())
    # people-low
    latest_deep_stats_fifth_people_low_improved = TextComp(
        latest_deep_fifth_people_low_improved, original_fifth_deep)
    print("[people_low_deepspeech_fifth_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_fifth_people_low_improved.WER(debug)))
    print("[people_low_deepspeech_fifth_2021_improved] Accuracy:" +
          str(latest_deep_stats_fifth_people_low_improved.Accuracy()))
    people_low_deep_accuracy_arr.append(
        latest_deep_stats_fifth_people_low_improved.Accuracy())
    # people-medium
    latest_deep_stats_fifth_people_medium_improved = TextComp(
        latest_deep_fifth_people_medium_improved, original_fifth_deep)
    print("_people_medium_deepspeech_fifth_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_fifth_people_medium_improved.WER(debug)))
    print("[people_medium_deepspeech_fifth_2021_improved] Accuracy:" +
          str(latest_deep_stats_fifth_people_medium_improved.Accuracy()))
    people_medium_deep_accuracy_arr.append(
        latest_deep_stats_fifth_people_medium_improved.Accuracy())
    # people-high
    latest_deep_stats_fifth_people_high_improved = TextComp(
        latest_deep_fifth_people_high_improved, original_fifth_deep)
    print("[people_high_deepspeech_fifth_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_fifth_people_high_improved.WER(debug)))
    print("[people_high_deepspeech_fifth_2021_improved] Accuracy:" +
          str(latest_deep_stats_fifth_people_high_improved.Accuracy()))
    people_high_deep_accuracy_arr.append(
        latest_deep_stats_fifth_people_high_improved.Accuracy())
    # sirens-low
    latest_deep_stats_fifth_sirens_low_improved = TextComp(
        latest_deep_fifth_sirens_low_improved, original_fifth_deep)
    print("[sirens_low_deepspeech_fifth_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_fifth_sirens_low_improved.WER(debug)))
    print("[sirens_low_deepspeech_fifth_2021_improved] Accuracy:" +
          str(latest_deep_stats_fifth_sirens_low_improved.Accuracy()))
    sirens_low_deep_accuracy_arr.append(
        latest_deep_stats_fifth_sirens_low_improved.Accuracy())
    # sirens-medium
    latest_deep_stats_fifth_sirens_medium_improved = TextComp(
        latest_deep_fifth_sirens_medium_improved, original_fifth_deep)
    print("[sirens_medium_deepspeech_fifth_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_fifth_sirens_medium_improved.WER(debug)))
    print("[sirens_medium_deepspeech_fifth_2021_improved] Accuracy:" +
          str(latest_deep_stats_fifth_sirens_medium_improved.Accuracy()))
    sirens_medium_deep_accuracy_arr.append(
        latest_deep_stats_fifth_sirens_medium_improved.Accuracy())
    # sirens-high
    latest_deep_stats_fifth_sirens_high_improved = TextComp(
        latest_deep_fifth_sirens_high_improved, original_fifth_deep)
    print("[sirens_high_deepspeech_fifth_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_fifth_sirens_high_improved.WER(debug)))
    print("[sirens_high_deepspeech_fifth_2021_improved] Accuracy:" +
          str(latest_deep_stats_fifth_sirens_high_improved.Accuracy()))
    sirens_high_deep_accuracy_arr.append(
        latest_deep_stats_fifth_sirens_high_improved.Accuracy())

    # sixth file
    ### Old Version of DeepSpeech ###
    # clean
    latest_deep_stats_sixth = TextComp(latest_deep_sixth, original_sixth_deep)
    print("[clean_deepspeech_sixth_2019] Word Error Rate:" +
          str(latest_deep_stats_sixth.WER(debug)))
    print("[clean_deepspeech_sixth_2019] Accuracy:" +
          str(latest_deep_stats_sixth.Accuracy()))
    clean_deep_accuracy_arr.append(latest_deep_stats_sixth.Accuracy())
    # cafe-low
    latest_deep_stats_sixth_cafe_low = TextComp(
        latest_deep_sixth_cafe_low, original_sixth_deep)
    print("[cafe_low_deepspeech_sixth_2019] Word Error Rate:" +
          str(latest_deep_stats_sixth_cafe_low.WER(debug)))
    print("[cafe_low_deepspeech_sixth_2019] Accuracy:" +
          str(latest_deep_stats_sixth_cafe_low.Accuracy()))
    cafe_low_deep_accuracy_arr.append(
        latest_deep_stats_sixth_cafe_low.Accuracy())
    # cafe-medium
    latest_deep_stats_sixth_cafe_medium = TextComp(
        latest_deep_sixth_cafe_medium, original_sixth_deep)
    print("[cafe_medium_deepspeech_sixth_2019] Word Error Rate:" +
          str(latest_deep_stats_sixth_cafe_medium.WER(debug)))
    print("[cafe_medium_deepspeech_sixth_2019] Accuracy:" +
          str(latest_deep_stats_sixth_cafe_medium.Accuracy()))
    cafe_medium_deep_accuracy_arr.append(
        latest_deep_stats_sixth_cafe_medium.Accuracy())
    # cafe-high
    latest_deep_stats_sixth_cafe_high = TextComp(
        latest_deep_sixth_cafe_high, original_sixth_deep)
    print("[cafe_high_deepspeech_sixth_2019] Word Error Rate:" +
          str(latest_deep_stats_sixth_cafe_high.WER(debug)))
    print("[cafe_high_deepspeech_sixth_2019] Accuracy:" +
          str(latest_deep_stats_sixth_cafe_high.Accuracy()))
    cafe_high_deep_accuracy_arr.append(
        latest_deep_stats_sixth_cafe_high.Accuracy())
    # people-low
    latest_deep_stats_sixth_people_low = TextComp(
        latest_deep_sixth_people_low, original_sixth_deep)
    print("[people_low_deepspeech_sixth_2019] Word Error Rate:" +
          str(latest_deep_stats_sixth_people_low.WER(debug)))
    print("[people_low_deepspeech_sixth_2019] Accuracy:" +
          str(latest_deep_stats_sixth_people_low.Accuracy()))
    people_low_deep_accuracy_arr.append(
        latest_deep_stats_sixth_people_low.Accuracy())
    # people-medium
    latest_deep_stats_sixth_people_medium = TextComp(
        latest_deep_sixth_people_medium, original_sixth_deep)
    print("[people_medium_deepspeech_sixth_2019] Word Error Rate:" +
          str(latest_deep_stats_sixth_people_medium.WER(debug)))
    print("[people_medium_deepspeech_sixth_2019] Accuracy:" +
          str(latest_deep_stats_sixth_people_medium.Accuracy()))
    people_medium_deep_accuracy_arr.append(
        latest_deep_stats_sixth_people_medium.Accuracy())
    # people-high
    latest_deep_stats_sixth_people_high = TextComp(
        latest_deep_sixth_people_high, original_sixth_deep)
    print("[people_high_deepspeech_sixth_2019] Word Error Rate:" +
          str(latest_deep_stats_sixth_people_high.WER(debug)))
    print("[people_high_deepspeech_sixth_2019] Accuracy:" +
          str(latest_deep_stats_sixth_people_high.Accuracy()))
    people_high_deep_accuracy_arr.append(
        latest_deep_stats_sixth_people_high.Accuracy())
    # sirens-low
    latest_deep_stats_sixth_sirens_low = TextComp(
        latest_deep_sixth_sirens_low, original_sixth_deep)
    print("[sirens_low_deepspeech_sixth_2019] Word Error Rate:" +
          str(latest_deep_stats_sixth_sirens_low.WER(debug)))
    print("[sirens_low_deepspeech_sixth_2019] Accuracy:" +
          str(latest_deep_stats_sixth_sirens_low.Accuracy()))
    sirens_low_deep_accuracy_arr.append(
        latest_deep_stats_sixth_sirens_low.Accuracy())
    # sirens-medium
    latest_deep_stats_sixth_sirens_medium = TextComp(
        latest_deep_sixth_sirens_medium, original_sixth_deep)
    print("[sirens_medium_deepspeech_sixth_2019] Word Error Rate:" +
          str(latest_deep_stats_sixth_sirens_medium.WER(debug)))
    print("[sirens_medium_deepspeech_sixth_2019] Accuracy:" +
          str(latest_deep_stats_sixth_sirens_medium.Accuracy()))
    sirens_medium_deep_accuracy_arr.append(
        latest_deep_stats_sixth_sirens_medium.Accuracy())
    # sirens-high
    latest_deep_stats_sixth_sirens_high = TextComp(
        latest_deep_sixth_sirens_high, original_sixth_deep)
    print("[sirens_high_deepspeech_sixth_2019] Word Error Rate:" +
          str(latest_deep_stats_sixth_sirens_high.WER(debug)))
    print("[sirens_high_deepspeech_sixth_2019] Accuracy:" +
          str(latest_deep_stats_sixth_sirens_high.Accuracy()))
    sirens_high_deep_accuracy_arr.append(
        latest_deep_stats_sixth_sirens_high.Accuracy())

    ### New version of DeepSpeech###
    # clean
    latest_deep_stats_sixth_new = TextComp(
        latest_deep_sixth_new, original_sixth_deep)
    print("[clean_deepspeech_sixth_2021] Word Error Rate:" +
          str(latest_deep_stats_sixth_new.WER(debug)))
    print("[clean_deepspeech_sixth_2021] Accuracy:" +
          str(latest_deep_stats_sixth_new.Accuracy()))
    clean_deep_accuracy_arr.append(latest_deep_stats_sixth_new.Accuracy())
    # cafe-low
    latest_deep_stats_sixth_cafe_low_new = TextComp(
        latest_deep_sixth_cafe_low_new, original_sixth_deep)
    print("[cafe_low_deepspeech_sixth_2021] Word Error Rate:" +
          str(latest_deep_stats_sixth_cafe_low_new.WER(debug)))
    print("[cafe_low_deepspeech_sixth_2021] Accuracy:" +
          str(latest_deep_stats_sixth_cafe_low_new.Accuracy()))
    cafe_low_deep_accuracy_arr.append(
        latest_deep_stats_sixth_cafe_low_new.Accuracy())
    # cafe-medium
    latest_deep_stats_sixth_cafe_medium_new = TextComp(
        latest_deep_sixth_cafe_medium_new, original_sixth_deep)
    print("[cafe_medium_deepspeech_sixth_2021] Word Error Rate:" +
          str(latest_deep_stats_sixth_cafe_medium_new.WER(debug)))
    print("[cafe_medium_deepspeech_sixth_2021] Accuracy:" +
          str(latest_deep_stats_sixth_cafe_medium_new.Accuracy()))
    cafe_medium_deep_accuracy_arr.append(
        latest_deep_stats_sixth_cafe_medium_new.Accuracy())
    # cafe-high
    latest_deep_stats_sixth_cafe_high_new = TextComp(
        latest_deep_sixth_cafe_high_new, original_sixth_deep)
    print("[cafe_high_deepspeech_sixth_2021] Word Error Rate:" +
          str(latest_deep_stats_sixth_cafe_high_new.WER(debug)))
    print("[cafe_high_deepspeech_sixth_2021] Accuracy:" +
          str(latest_deep_stats_sixth_cafe_high_new.Accuracy()))
    cafe_high_deep_accuracy_arr.append(
        latest_deep_stats_sixth_cafe_high_new.Accuracy())
    # people-low
    latest_deep_stats_sixth_people_low_new = TextComp(
        latest_deep_sixth_people_low_new, original_sixth_deep)
    print("[people_low_deepspeech_sixth_2021] Word Error Rate:" +
          str(latest_deep_stats_sixth_people_low_new.WER(debug)))
    print("[people_low_deepspeech_sixth_2021] Accuracy:" +
          str(latest_deep_stats_sixth_people_low_new.Accuracy()))
    people_low_deep_accuracy_arr.append(
        latest_deep_stats_sixth_people_low_new.Accuracy())
    # people-medium
    latest_deep_stats_sixth_people_medium_new = TextComp(
        latest_deep_sixth_people_medium_new, original_sixth_deep)
    print("_people_medium_deepspeech_sixth_2021] Word Error Rate:" +
          str(latest_deep_stats_sixth_people_medium_new.WER(debug)))
    print("[people_medium_deepspeech_sixth_2021] Accuracy:" +
          str(latest_deep_stats_sixth_people_medium_new.Accuracy()))
    people_medium_deep_accuracy_arr.append(
        latest_deep_stats_sixth_people_medium_new.Accuracy())
    # people-high
    latest_deep_stats_sixth_people_high_new = TextComp(
        latest_deep_sixth_people_high_new, original_sixth_deep)
    print("[people_high_deepspeech_sixth_2021] Word Error Rate:" +
          str(latest_deep_stats_sixth_people_high_new.WER(debug)))
    print("[people_high_deepspeech_sixth_2021] Accuracy:" +
          str(latest_deep_stats_sixth_people_high_new.Accuracy()))
    people_high_deep_accuracy_arr.append(
        latest_deep_stats_sixth_people_high_new.Accuracy())
    # sirens-low
    latest_deep_stats_sixth_sirens_low_new = TextComp(
        latest_deep_sixth_sirens_low_new, original_sixth_deep)
    print("[sirens_low_deepspeech_sixth_2021] Word Error Rate:" +
          str(latest_deep_stats_sixth_sirens_low_new.WER(debug)))
    print("[sirens_low_deepspeech_sixth_2021] Accuracy:" +
          str(latest_deep_stats_sixth_sirens_low_new.Accuracy()))
    sirens_low_deep_accuracy_arr.append(
        latest_deep_stats_sixth_sirens_low_new.Accuracy())
    # sirens-medium
    latest_deep_stats_sixth_sirens_medium_new = TextComp(
        latest_deep_sixth_sirens_medium_new, original_sixth_deep)
    print("[sirens_medium_deepspeech_sixth_2021] Word Error Rate:" +
          str(latest_deep_stats_sixth_sirens_medium_new.WER(debug)))
    print("[sirens_medium_deepspeech_sixth_2021] Accuracy:" +
          str(latest_deep_stats_sixth_sirens_medium_new.Accuracy()))
    sirens_medium_deep_accuracy_arr.append(
        latest_deep_stats_sixth_sirens_medium_new.Accuracy())
    # sirens-high
    latest_deep_stats_sixth_sirens_high_new = TextComp(
        latest_deep_sixth_sirens_high_new, original_sixth_deep)
    print("[sirens_high_deepspeech_sixth_2021] Word Error Rate:" +
          str(latest_deep_stats_sixth_sirens_high_new.WER(debug)))
    print("[sirens_high_deepspeech_sixth_2021] Accuracy:" +
          str(latest_deep_stats_sixth_sirens_high_new.Accuracy()))
    sirens_high_deep_accuracy_arr.append(
        latest_deep_stats_sixth_sirens_high_new.Accuracy())

     ### New version of DeepSpeech with improved scorer###
    # clean
    latest_deep_stats_sixth_improved = TextComp(
        latest_deep_sixth_improved, original_sixth_deep)
    print("[clean_deepspeech_sixth_2021_imp] Word Error Rate:" +
          str(latest_deep_stats_sixth_improved.WER(debug)))
    print("[clean_deepspeech_sixth_2021_imp] Accuracy:" +
          str(latest_deep_stats_sixth_improved.Accuracy()))
    clean_deep_accuracy_arr.append(latest_deep_stats_sixth_improved.Accuracy())
    # cafe-low
    latest_deep_stats_sixth_cafe_low_improved = TextComp(
        latest_deep_sixth_cafe_low_improved, original_sixth_deep)
    print("[cafe_low_deepspeech_sixth_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_sixth_cafe_low_improved.WER(debug)))
    print("[cafe_low_deepspeech_sixth_2021_improved] Accuracy:" +
          str(latest_deep_stats_sixth_cafe_low_improved.Accuracy()))
    cafe_low_deep_accuracy_arr.append(
        latest_deep_stats_sixth_cafe_low_improved.Accuracy())
    # cafe-medium
    latest_deep_stats_sixth_cafe_medium_improved = TextComp(
        latest_deep_sixth_cafe_medium_improved, original_sixth_deep)
    print("[cafe_medium_deepspeech_sixth_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_sixth_cafe_medium_improved.WER(debug)))
    print("[cafe_medium_deepspeech_sixth_2021_improved] Accuracy:" +
          str(latest_deep_stats_sixth_cafe_medium_improved.Accuracy()))
    cafe_medium_deep_accuracy_arr.append(
        latest_deep_stats_sixth_cafe_medium_improved.Accuracy())
    # cafe-high
    latest_deep_stats_sixth_cafe_high_improved = TextComp(
        latest_deep_sixth_cafe_high_improved, original_sixth_deep)
    print("[cafe_high_deepspeech_sixth_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_sixth_cafe_high_improved.WER(debug)))
    print("[cafe_high_deepspeech_sixth_2021_improved] Accuracy:" +
          str(latest_deep_stats_sixth_cafe_high_improved.Accuracy()))
    cafe_high_deep_accuracy_arr.append(
        latest_deep_stats_sixth_cafe_high_improved.Accuracy())
    # people-low
    latest_deep_stats_sixth_people_low_improved = TextComp(
        latest_deep_sixth_people_low_improved, original_sixth_deep)
    print("[people_low_deepspeech_sixth_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_sixth_people_low_improved.WER(debug)))
    print("[people_low_deepspeech_sixth_2021_improved] Accuracy:" +
          str(latest_deep_stats_sixth_people_low_improved.Accuracy()))
    people_low_deep_accuracy_arr.append(
        latest_deep_stats_sixth_people_low_improved.Accuracy())
    # people-medium
    latest_deep_stats_sixth_people_medium_improved = TextComp(
        latest_deep_sixth_people_medium_improved, original_sixth_deep)
    print("_people_medium_deepspeech_sixth_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_sixth_people_medium_improved.WER(debug)))
    print("[people_medium_deepspeech_sixth_2021_improved] Accuracy:" +
          str(latest_deep_stats_sixth_people_medium_improved.Accuracy()))
    people_medium_deep_accuracy_arr.append(
        latest_deep_stats_sixth_people_medium_improved.Accuracy())
    # people-high
    latest_deep_stats_sixth_people_high_improved = TextComp(
        latest_deep_sixth_people_high_improved, original_sixth_deep)
    print("[people_high_deepspeech_sixth_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_sixth_people_high_improved.WER(debug)))
    print("[people_high_deepspeech_sixth_2021_improved] Accuracy:" +
          str(latest_deep_stats_sixth_people_high_improved.Accuracy()))
    people_high_deep_accuracy_arr.append(
        latest_deep_stats_sixth_people_high_improved.Accuracy())
    # sirens-low
    latest_deep_stats_sixth_sirens_low_improved = TextComp(
        latest_deep_sixth_sirens_low_improved, original_sixth_deep)
    print("[sirens_low_deepspeech_sixth_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_sixth_sirens_low_improved.WER(debug)))
    print("[sirens_low_deepspeech_sixth_2021_improved] Accuracy:" +
          str(latest_deep_stats_sixth_sirens_low_improved.Accuracy()))
    sirens_low_deep_accuracy_arr.append(
        latest_deep_stats_sixth_sirens_low_improved.Accuracy())
    # sirens-medium
    latest_deep_stats_sixth_sirens_medium_improved = TextComp(
        latest_deep_sixth_sirens_medium_improved, original_sixth_deep)
    print("[sirens_medium_deepspeech_sixth_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_sixth_sirens_medium_improved.WER(debug)))
    print("[sirens_medium_deepspeech_sixth_2021_improved] Accuracy:" +
          str(latest_deep_stats_sixth_sirens_medium_improved.Accuracy()))
    sirens_medium_deep_accuracy_arr.append(
        latest_deep_stats_sixth_sirens_medium_improved.Accuracy())
    # sirens-high
    latest_deep_stats_sixth_sirens_high_improved = TextComp(
        latest_deep_sixth_sirens_high_improved, original_sixth_deep)
    print("[sirens_high_deepspeech_sixth_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_sixth_sirens_high_improved.WER(debug)))
    print("[sirens_high_deepspeech_sixth_2021_improved] Accuracy:" +
          str(latest_deep_stats_sixth_sirens_high_improved.Accuracy()))
    sirens_high_deep_accuracy_arr.append(
        latest_deep_stats_sixth_sirens_high_improved.Accuracy())
    
    # seventh file
    ### Old Version of DeepSpeech ###
    # clean
    latest_deep_stats_seventh = TextComp(
        latest_deep_seventh, original_seventh_deep)
    print("[clean_deepspeech_seventh_2019] Word Error Rate:" +
          str(latest_deep_stats_seventh.WER(debug)))
    print("[clean_deepspeech_seventh_2019] Accuracy:" +
          str(latest_deep_stats_seventh.Accuracy()))
    clean_deep_accuracy_arr.append(latest_deep_stats_seventh.Accuracy())
    # cafe-low
    latest_deep_stats_seventh_cafe_low = TextComp(
        latest_deep_seventh_cafe_low, original_seventh_deep)
    print("[cafe_low_deepspeech_seventh_2019] Word Error Rate:" +
          str(latest_deep_stats_seventh_cafe_low.WER(debug)))
    print("[cafe_low_deepspeech_seventh_2019] Accuracy:" +
          str(latest_deep_stats_seventh_cafe_low.Accuracy()))
    cafe_low_deep_accuracy_arr.append(
        latest_deep_stats_seventh_cafe_low.Accuracy())
    # cafe-medium
    latest_deep_stats_seventh_cafe_medium = TextComp(
        latest_deep_seventh_cafe_medium, original_seventh_deep)
    print("[cafe_medium_deepspeech_seventh_2019] Word Error Rate:" +
          str(latest_deep_stats_seventh_cafe_medium.WER(debug)))
    print("[cafe_medium_deepspeech_seventh_2019] Accuracy:" +
          str(latest_deep_stats_seventh_cafe_medium.Accuracy()))
    cafe_medium_deep_accuracy_arr.append(
        latest_deep_stats_seventh_cafe_medium.Accuracy())
    # cafe-high
    latest_deep_stats_seventh_cafe_high = TextComp(
        latest_deep_seventh_cafe_high, original_seventh_deep)
    print("[cafe_high_deepspeech_seventh_2019] Word Error Rate:" +
          str(latest_deep_stats_seventh_cafe_high.WER(debug)))
    print("[cafe_high_deepspeech_seventh_2019] Accuracy:" +
          str(latest_deep_stats_seventh_cafe_high.Accuracy()))
    cafe_high_deep_accuracy_arr.append(
        latest_deep_stats_seventh_cafe_high.Accuracy())
    # people-low
    latest_deep_stats_seventh_people_low = TextComp(
        latest_deep_seventh_people_low, original_seventh_deep)
    print("[people_low_deepspeech_seventh_2019] Word Error Rate:" +
          str(latest_deep_stats_seventh_people_low.WER(debug)))
    print("[people_low_deepspeech_seventh_2019] Accuracy:" +
          str(latest_deep_stats_seventh_people_low.Accuracy()))
    people_low_deep_accuracy_arr.append(
        latest_deep_stats_seventh_people_low.Accuracy())
    # people-medium
    latest_deep_stats_seventh_people_medium = TextComp(
        latest_deep_seventh_people_medium, original_seventh_deep)
    print("[people_medium_deepspeech_seventh_2019] Word Error Rate:" +
          str(latest_deep_stats_seventh_people_medium.WER(debug)))
    print("[people_medium_deepspeech_seventh_2019] Accuracy:" +
          str(latest_deep_stats_seventh_people_medium.Accuracy()))
    people_medium_deep_accuracy_arr.append(
        latest_deep_stats_seventh_people_medium.Accuracy())
    # people-high
    latest_deep_stats_seventh_people_high = TextComp(
        latest_deep_seventh_people_high, original_seventh_deep)
    print("[people_high_deepspeech_seventh_2019] Word Error Rate:" +
          str(latest_deep_stats_seventh_people_high.WER(debug)))
    print("[people_high_deepspeech_seventh_2019] Accuracy:" +
          str(latest_deep_stats_seventh_people_high.Accuracy()))
    people_high_deep_accuracy_arr.append(
        latest_deep_stats_seventh_people_high.Accuracy())
    # sirens-low
    latest_deep_stats_seventh_sirens_low = TextComp(
        latest_deep_seventh_sirens_low, original_seventh_deep)
    print("[sirens_low_deepspeech_seventh_2019] Word Error Rate:" +
          str(latest_deep_stats_seventh_sirens_low.WER(debug)))
    print("[sirens_low_deepspeech_seventh_2019] Accuracy:" +
          str(latest_deep_stats_seventh_sirens_low.Accuracy()))
    sirens_low_deep_accuracy_arr.append(
        latest_deep_stats_seventh_sirens_low.Accuracy())
    # sirens-medium
    latest_deep_stats_seventh_sirens_medium = TextComp(
        latest_deep_seventh_sirens_medium, original_seventh_deep)
    print("[sirens_medium_deepspeech_seventh_2019] Word Error Rate:" +
          str(latest_deep_stats_seventh_sirens_medium.WER(debug)))
    print("[sirens_medium_deepspeech_seventh_2019] Accuracy:" +
          str(latest_deep_stats_seventh_sirens_medium.Accuracy()))
    sirens_medium_deep_accuracy_arr.append(
        latest_deep_stats_seventh_sirens_medium.Accuracy())
    # sirens-high
    latest_deep_stats_seventh_sirens_high = TextComp(
        latest_deep_seventh_sirens_high, original_seventh_deep)
    print("[sirens_high_deepspeech_seventh_2019] Word Error Rate:" +
          str(latest_deep_stats_seventh_sirens_high.WER(debug)))
    print("[sirens_high_deepspeech_seventh_2019] Accuracy:" +
          str(latest_deep_stats_seventh_sirens_high.Accuracy()))
    sirens_high_deep_accuracy_arr.append(
        latest_deep_stats_seventh_sirens_high.Accuracy())

    ### New Version of DeepSpeech ###
    # clean
    latest_deep_stats_seventh_new = TextComp(
        latest_deep_seventh_new, original_seventh_deep)
    print("[clean_deepspeech_seventh_2021] Word Error Rate:" +
          str(latest_deep_stats_seventh_new.WER(debug)))
    print("[clean_deepspeech_seventh_2021] Accuracy:" +
          str(latest_deep_stats_seventh_new.Accuracy()))
    clean_deep_accuracy_arr.append(latest_deep_stats_seventh_new.Accuracy())
    # cafe-low
    latest_deep_stats_seventh_cafe_low_new = TextComp(
        latest_deep_seventh_cafe_low_new, original_seventh_deep)
    print("[cafe_low_deepspeech_seventh_2021] Word Error Rate:" +
          str(latest_deep_stats_seventh_cafe_low_new.WER(debug)))
    print("[cafe_low_deepspeech_seventh_2021] Accuracy:" +
          str(latest_deep_stats_seventh_cafe_low_new.Accuracy()))
    cafe_low_deep_accuracy_arr.append(
        latest_deep_stats_seventh_cafe_low_new.Accuracy())
    # cafe-medium
    latest_deep_stats_seventh_cafe_medium_new = TextComp(
        latest_deep_seventh_cafe_medium_new, original_seventh_deep)
    print("[cafe_medium_deepspeech_seventh_2021] Word Error Rate:" +
          str(latest_deep_stats_seventh_cafe_medium_new.WER(debug)))
    print("[cafe_medium_deepspeech_seventh_2021] Accuracy:" +
          str(latest_deep_stats_seventh_cafe_medium_new.Accuracy()))
    cafe_medium_deep_accuracy_arr.append(
        latest_deep_stats_seventh_cafe_medium_new.Accuracy())
    # cafe-high
    latest_deep_stats_seventh_cafe_high_new = TextComp(
        latest_deep_seventh_cafe_high_new, original_seventh_deep)
    print("[cafe_high_deepspeech_seventh_2021] Word Error Rate:" +
          str(latest_deep_stats_seventh_cafe_high_new.WER(debug)))
    print("[cafe_high_deepspeech_seventh_2021] Accuracy:" +
          str(latest_deep_stats_seventh_cafe_high_new.Accuracy()))
    cafe_high_deep_accuracy_arr.append(
        latest_deep_stats_seventh_cafe_high_new.Accuracy())
    # people-low
    latest_deep_stats_seventh_people_low_new = TextComp(
        latest_deep_seventh_people_low_new, original_seventh_deep)
    print("[people_low_deepspeech_seventh_2021] Word Error Rate:" +
          str(latest_deep_stats_seventh_people_low_new.WER(debug)))
    print("[people_low_deepspeech_seventh_2021] Accuracy:" +
          str(latest_deep_stats_seventh_people_low_new.Accuracy()))
    people_low_deep_accuracy_arr.append(
        latest_deep_stats_seventh_people_low_new.Accuracy())
    # people-medium
    latest_deep_stats_seventh_people_medium_new = TextComp(
        latest_deep_seventh_people_medium_new, original_seventh_deep)
    print("[people_medium_deepspeech_seventh_2021] Word Error Rate:" +
          str(latest_deep_stats_seventh_people_medium_new.WER(debug)))
    print("[people_medium_deepspeech_seventh_2021] Accuracy:" +
          str(latest_deep_stats_seventh_people_medium_new.Accuracy()))
    people_medium_deep_accuracy_arr.append(
        latest_deep_stats_seventh_people_medium_new.Accuracy())
    # people-high
    latest_deep_stats_seventh_people_high_new = TextComp(
        latest_deep_seventh_people_high_new, original_seventh_deep)
    print("[people_high_deepspeech_seventh_2021] Word Error Rate:" +
          str(latest_deep_stats_seventh_people_high_new.WER(debug)))
    print("[people_high_deepspeech_seventh_2021] Accuracy:" +
          str(latest_deep_stats_seventh_people_high_new.Accuracy()))
    people_high_deep_accuracy_arr.append(
        latest_deep_stats_seventh_people_high_new.Accuracy())
    # sirens-low
    latest_deep_stats_seventh_sirens_low_new = TextComp(
        latest_deep_seventh_sirens_low_new, original_seventh_deep)
    print("[sirens_low_deepspeech_seventh_2021] Word Error Rate:" +
          str(latest_deep_stats_seventh_sirens_low_new.WER(debug)))
    print("[sirens_low_deepspeech_seventh_2021] Accuracy:" +
          str(latest_deep_stats_seventh_sirens_low_new.Accuracy()))
    sirens_low_deep_accuracy_arr.append(
        latest_deep_stats_seventh_sirens_low_new.Accuracy())
    # sirens-medium
    latest_deep_stats_seventh_sirens_medium_new = TextComp(
        latest_deep_seventh_sirens_medium_new, original_seventh_deep)
    print("[sirens_medium_deepspeech_seventh_2021] Word Error Rate:" +
          str(latest_deep_stats_seventh_sirens_medium_new.WER(debug)))
    print("[sirens_medium_deepspeech_seventh_2021] Accuracy:" +
          str(latest_deep_stats_seventh_sirens_medium_new.Accuracy()))
    sirens_medium_deep_accuracy_arr.append(
        latest_deep_stats_seventh_sirens_medium_new.Accuracy())
    # sirens-high
    latest_deep_stats_seventh_sirens_high_new = TextComp(
        latest_deep_seventh_sirens_high_new, original_seventh_deep)
    print("[sirens_high_deepspeech_seventh_2021] Word Error Rate:" +
          str(latest_deep_stats_seventh_sirens_high_new.WER(debug)))
    print("[sirens_high_deepspeech_seventh_2021] Accuracy:" +
          str(latest_deep_stats_seventh_sirens_high_new.Accuracy()))
    sirens_high_deep_accuracy_arr.append(
        latest_deep_stats_seventh_sirens_high_new.Accuracy())
    
     ### New version of DeepSpeech with improved scorer###
    # clean
    latest_deep_stats_seventh_improved = TextComp(
        latest_deep_seventh_improved, original_seventh_deep)
    print("[clean_deepspeech_seventh_2021_imp] Word Error Rate:" +
          str(latest_deep_stats_seventh_improved.WER(debug)))
    print("[clean_deepspeech_seventh_2021_imp] Accuracy:" +
          str(latest_deep_stats_seventh_improved.Accuracy()))
    clean_deep_accuracy_arr.append(latest_deep_stats_seventh_improved.Accuracy())
    # cafe-low
    latest_deep_stats_seventh_cafe_low_improved = TextComp(
        latest_deep_seventh_cafe_low_improved, original_seventh_deep)
    print("[cafe_low_deepspeech_seventh_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_seventh_cafe_low_improved.WER(debug)))
    print("[cafe_low_deepspeech_seventh_2021_improved] Accuracy:" +
          str(latest_deep_stats_seventh_cafe_low_improved.Accuracy()))
    cafe_low_deep_accuracy_arr.append(
        latest_deep_stats_seventh_cafe_low_improved.Accuracy())
    # cafe-medium
    latest_deep_stats_seventh_cafe_medium_improved = TextComp(
        latest_deep_seventh_cafe_medium_improved, original_seventh_deep)
    print("[cafe_medium_deepspeech_seventh_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_seventh_cafe_medium_improved.WER(debug)))
    print("[cafe_medium_deepspeech_seventh_2021_improved] Accuracy:" +
          str(latest_deep_stats_seventh_cafe_medium_improved.Accuracy()))
    cafe_medium_deep_accuracy_arr.append(
        latest_deep_stats_seventh_cafe_medium_improved.Accuracy())
    # cafe-high
    latest_deep_stats_seventh_cafe_high_improved = TextComp(
        latest_deep_seventh_cafe_high_improved, original_seventh_deep)
    print("[cafe_high_deepspeech_seventh_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_seventh_cafe_high_improved.WER(debug)))
    print("[cafe_high_deepspeech_seventh_2021_improved] Accuracy:" +
          str(latest_deep_stats_seventh_cafe_high_improved.Accuracy()))
    cafe_high_deep_accuracy_arr.append(
        latest_deep_stats_seventh_cafe_high_improved.Accuracy())
    # people-low
    latest_deep_stats_seventh_people_low_improved = TextComp(
        latest_deep_seventh_people_low_improved, original_seventh_deep)
    print("[people_low_deepspeech_seventh_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_seventh_people_low_improved.WER(debug)))
    print("[people_low_deepspeech_seventh_2021_improved] Accuracy:" +
          str(latest_deep_stats_seventh_people_low_improved.Accuracy()))
    people_low_deep_accuracy_arr.append(
        latest_deep_stats_seventh_people_low_improved.Accuracy())
    # people-medium
    latest_deep_stats_seventh_people_medium_improved = TextComp(
        latest_deep_seventh_people_medium_improved, original_seventh_deep)
    print("_people_medium_deepspeech_seventh_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_seventh_people_medium_improved.WER(debug)))
    print("[people_medium_deepspeech_seventh_2021_improved] Accuracy:" +
          str(latest_deep_stats_seventh_people_medium_improved.Accuracy()))
    people_medium_deep_accuracy_arr.append(
        latest_deep_stats_seventh_people_medium_improved.Accuracy())
    # people-high
    latest_deep_stats_seventh_people_high_improved = TextComp(
        latest_deep_seventh_people_high_improved, original_seventh_deep)
    print("[people_high_deepspeech_seventh_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_seventh_people_high_improved.WER(debug)))
    print("[people_high_deepspeech_seventh_2021_improved] Accuracy:" +
          str(latest_deep_stats_seventh_people_high_improved.Accuracy()))
    people_high_deep_accuracy_arr.append(
        latest_deep_stats_seventh_people_high_improved.Accuracy())
    # sirens-low
    latest_deep_stats_seventh_sirens_low_improved = TextComp(
        latest_deep_seventh_sirens_low_improved, original_seventh_deep)
    print("[sirens_low_deepspeech_seventh_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_seventh_sirens_low_improved.WER(debug)))
    print("[sirens_low_deepspeech_seventh_2021_improved] Accuracy:" +
          str(latest_deep_stats_seventh_sirens_low_improved.Accuracy()))
    sirens_low_deep_accuracy_arr.append(
        latest_deep_stats_seventh_sirens_low_improved.Accuracy())
    # sirens-medium
    latest_deep_stats_seventh_sirens_medium_improved = TextComp(
        latest_deep_seventh_sirens_medium_improved, original_seventh_deep)
    print("[sirens_medium_deepspeech_seventh_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_seventh_sirens_medium_improved.WER(debug)))
    print("[sirens_medium_deepspeech_seventh_2021_improved] Accuracy:" +
          str(latest_deep_stats_seventh_sirens_medium_improved.Accuracy()))
    sirens_medium_deep_accuracy_arr.append(
        latest_deep_stats_seventh_sirens_medium_improved.Accuracy())
    # sirens-high
    latest_deep_stats_seventh_sirens_high_improved = TextComp(
        latest_deep_seventh_sirens_high_improved, original_seventh_deep)
    print("[sirens_high_deepspeech_seventh_2021_improved] Word Error Rate:" +
          str(latest_deep_stats_seventh_sirens_high_improved.WER(debug)))
    print("[sirens_high_deepspeech_seventh_2021_improved] Accuracy:" +
          str(latest_deep_stats_seventh_sirens_high_improved.Accuracy()))
    sirens_high_deep_accuracy_arr.append(
        latest_deep_stats_seventh_sirens_high_improved.Accuracy())

    # cafe data comparison arrays
    cafe_arr_first = []
    cafe_arr_first.append(clean_deep_accuracy_arr[0])
    cafe_arr_first.append(clean_deep_accuracy_arr[1])
    cafe_arr_first.append(clean_deep_accuracy_arr[2])
    cafe_arr_first.append(cafe_low_deep_accuracy_arr[0])
    cafe_arr_first.append(cafe_low_deep_accuracy_arr[1])
    cafe_arr_first.append(cafe_low_deep_accuracy_arr[2])
    cafe_arr_first.append(cafe_medium_deep_accuracy_arr[0])
    cafe_arr_first.append(cafe_medium_deep_accuracy_arr[1])
    cafe_arr_first.append(cafe_medium_deep_accuracy_arr[2])
    cafe_arr_first.append(cafe_high_deep_accuracy_arr[0])
    cafe_arr_first.append(cafe_high_deep_accuracy_arr[1])
    cafe_arr_first.append(cafe_high_deep_accuracy_arr[2])

    cafe_arr_second = []
    cafe_arr_second.append(clean_deep_accuracy_arr[3])
    cafe_arr_second.append(clean_deep_accuracy_arr[4])
    cafe_arr_second.append(clean_deep_accuracy_arr[5])
    cafe_arr_second.append(cafe_low_deep_accuracy_arr[3])
    cafe_arr_second.append(cafe_low_deep_accuracy_arr[4])
    cafe_arr_second.append(cafe_low_deep_accuracy_arr[5])
    cafe_arr_second.append(cafe_medium_deep_accuracy_arr[3])
    cafe_arr_second.append(cafe_medium_deep_accuracy_arr[4])
    cafe_arr_second.append(cafe_medium_deep_accuracy_arr[5])
    cafe_arr_second.append(cafe_high_deep_accuracy_arr[3])
    cafe_arr_second.append(cafe_high_deep_accuracy_arr[4])
    cafe_arr_second.append(cafe_high_deep_accuracy_arr[5])

    cafe_arr_third = []
    cafe_arr_third.append(clean_deep_accuracy_arr[6])
    cafe_arr_third.append(clean_deep_accuracy_arr[7])
    cafe_arr_third.append(clean_deep_accuracy_arr[8])
    cafe_arr_third.append(cafe_low_deep_accuracy_arr[6])
    cafe_arr_third.append(cafe_low_deep_accuracy_arr[7])
    cafe_arr_third.append(cafe_low_deep_accuracy_arr[8])
    cafe_arr_third.append(cafe_medium_deep_accuracy_arr[6])
    cafe_arr_third.append(cafe_medium_deep_accuracy_arr[7])
    cafe_arr_third.append(cafe_medium_deep_accuracy_arr[8])
    cafe_arr_third.append(cafe_high_deep_accuracy_arr[6])
    cafe_arr_third.append(cafe_high_deep_accuracy_arr[7])
    cafe_arr_third.append(cafe_high_deep_accuracy_arr[8])

    cafe_arr_fourth = []
    cafe_arr_fourth.append(clean_deep_accuracy_arr[9])
    cafe_arr_fourth.append(clean_deep_accuracy_arr[10])
    cafe_arr_fourth.append(clean_deep_accuracy_arr[11])
    cafe_arr_fourth.append(cafe_low_deep_accuracy_arr[9])
    cafe_arr_fourth.append(cafe_low_deep_accuracy_arr[10])
    cafe_arr_fourth.append(cafe_low_deep_accuracy_arr[11])
    cafe_arr_fourth.append(cafe_medium_deep_accuracy_arr[9])
    cafe_arr_fourth.append(cafe_medium_deep_accuracy_arr[10])
    cafe_arr_fourth.append(cafe_medium_deep_accuracy_arr[11])
    cafe_arr_fourth.append(cafe_high_deep_accuracy_arr[9])
    cafe_arr_fourth.append(cafe_high_deep_accuracy_arr[10])
    cafe_arr_fourth.append(cafe_high_deep_accuracy_arr[11])

    # people data comparison arrays
    people_arr_first = []
    people_arr_first.append(clean_deep_accuracy_arr[0])
    people_arr_first.append(clean_deep_accuracy_arr[1])
    people_arr_first.append(clean_deep_accuracy_arr[2])
    people_arr_first.append(people_low_deep_accuracy_arr[0])
    people_arr_first.append(people_low_deep_accuracy_arr[1])
    people_arr_first.append(people_low_deep_accuracy_arr[2])
    people_arr_first.append(people_medium_deep_accuracy_arr[0])
    people_arr_first.append(people_medium_deep_accuracy_arr[1])
    people_arr_first.append(people_medium_deep_accuracy_arr[2])
    people_arr_first.append(people_high_deep_accuracy_arr[0])
    people_arr_first.append(people_high_deep_accuracy_arr[1])
    people_arr_first.append(people_high_deep_accuracy_arr[2])

    people_arr_second = []
    people_arr_second.append(clean_deep_accuracy_arr[3])
    people_arr_second.append(clean_deep_accuracy_arr[4])
    people_arr_second.append(clean_deep_accuracy_arr[5])
    people_arr_second.append(people_low_deep_accuracy_arr[3])
    people_arr_second.append(people_low_deep_accuracy_arr[4])
    people_arr_second.append(people_low_deep_accuracy_arr[5])
    people_arr_second.append(people_medium_deep_accuracy_arr[3])
    people_arr_second.append(people_medium_deep_accuracy_arr[4])
    people_arr_second.append(people_medium_deep_accuracy_arr[5])
    people_arr_second.append(people_high_deep_accuracy_arr[3])
    people_arr_second.append(people_high_deep_accuracy_arr[4])
    people_arr_second.append(people_high_deep_accuracy_arr[5])

    people_arr_third = []
    people_arr_third.append(clean_deep_accuracy_arr[6])
    people_arr_third.append(clean_deep_accuracy_arr[7])
    people_arr_third.append(clean_deep_accuracy_arr[8])
    people_arr_third.append(people_low_deep_accuracy_arr[6])
    people_arr_third.append(people_low_deep_accuracy_arr[7])
    people_arr_third.append(people_low_deep_accuracy_arr[8])
    people_arr_third.append(people_medium_deep_accuracy_arr[6])
    people_arr_third.append(people_medium_deep_accuracy_arr[7])
    people_arr_third.append(people_medium_deep_accuracy_arr[8])
    people_arr_third.append(people_high_deep_accuracy_arr[6])
    people_arr_third.append(people_high_deep_accuracy_arr[7])
    people_arr_third.append(people_high_deep_accuracy_arr[8])

    people_arr_fourth = []
    people_arr_fourth.append(clean_deep_accuracy_arr[9])
    people_arr_fourth.append(clean_deep_accuracy_arr[10])
    people_arr_fourth.append(clean_deep_accuracy_arr[11])
    people_arr_fourth.append(people_low_deep_accuracy_arr[9])
    people_arr_fourth.append(people_low_deep_accuracy_arr[10])
    people_arr_fourth.append(people_low_deep_accuracy_arr[11])
    people_arr_fourth.append(people_medium_deep_accuracy_arr[9])
    people_arr_fourth.append(people_medium_deep_accuracy_arr[10])
    people_arr_fourth.append(people_medium_deep_accuracy_arr[11])
    people_arr_fourth.append(people_high_deep_accuracy_arr[9])
    people_arr_fourth.append(people_high_deep_accuracy_arr[10])
    people_arr_fourth.append(people_high_deep_accuracy_arr[11])

    # sirens data comparison arrays
    sirens_arr_first = []
    sirens_arr_first.append(clean_deep_accuracy_arr[0])
    sirens_arr_first.append(clean_deep_accuracy_arr[1])
    sirens_arr_first.append(clean_deep_accuracy_arr[2])
    sirens_arr_first.append(sirens_low_deep_accuracy_arr[0])
    sirens_arr_first.append(sirens_low_deep_accuracy_arr[1])
    sirens_arr_first.append(sirens_low_deep_accuracy_arr[2])
    sirens_arr_first.append(sirens_medium_deep_accuracy_arr[0])
    sirens_arr_first.append(sirens_medium_deep_accuracy_arr[1])
    sirens_arr_first.append(sirens_medium_deep_accuracy_arr[2])
    sirens_arr_first.append(sirens_high_deep_accuracy_arr[0])
    sirens_arr_first.append(sirens_high_deep_accuracy_arr[1])
    sirens_arr_first.append(sirens_high_deep_accuracy_arr[2])

    sirens_arr_second = []
    sirens_arr_second.append(clean_deep_accuracy_arr[3])
    sirens_arr_second.append(clean_deep_accuracy_arr[4])
    sirens_arr_second.append(clean_deep_accuracy_arr[5])
    sirens_arr_second.append(sirens_low_deep_accuracy_arr[3])
    sirens_arr_second.append(sirens_low_deep_accuracy_arr[4])
    sirens_arr_second.append(sirens_low_deep_accuracy_arr[5])
    sirens_arr_second.append(sirens_medium_deep_accuracy_arr[3])
    sirens_arr_second.append(sirens_medium_deep_accuracy_arr[4])
    sirens_arr_second.append(sirens_medium_deep_accuracy_arr[5])
    sirens_arr_second.append(sirens_high_deep_accuracy_arr[3])
    sirens_arr_second.append(sirens_high_deep_accuracy_arr[4])
    sirens_arr_second.append(sirens_high_deep_accuracy_arr[5])

    sirens_arr_third = []
    sirens_arr_third.append(clean_deep_accuracy_arr[6])
    sirens_arr_third.append(clean_deep_accuracy_arr[7])
    sirens_arr_third.append(clean_deep_accuracy_arr[8])
    sirens_arr_third.append(sirens_low_deep_accuracy_arr[6])
    sirens_arr_third.append(sirens_low_deep_accuracy_arr[7])
    sirens_arr_third.append(sirens_low_deep_accuracy_arr[8])
    sirens_arr_third.append(sirens_medium_deep_accuracy_arr[6])
    sirens_arr_third.append(sirens_medium_deep_accuracy_arr[7])
    sirens_arr_third.append(sirens_medium_deep_accuracy_arr[8])
    sirens_arr_third.append(sirens_high_deep_accuracy_arr[6])
    sirens_arr_third.append(sirens_high_deep_accuracy_arr[7])
    sirens_arr_third.append(sirens_high_deep_accuracy_arr[8])

    sirens_arr_fourth = []
    sirens_arr_fourth.append(clean_deep_accuracy_arr[9])
    sirens_arr_fourth.append(clean_deep_accuracy_arr[10])
    sirens_arr_fourth.append(clean_deep_accuracy_arr[11])
    sirens_arr_fourth.append(sirens_low_deep_accuracy_arr[9])
    sirens_arr_fourth.append(sirens_low_deep_accuracy_arr[10])
    sirens_arr_fourth.append(sirens_low_deep_accuracy_arr[11])
    sirens_arr_fourth.append(sirens_medium_deep_accuracy_arr[9])
    sirens_arr_fourth.append(sirens_medium_deep_accuracy_arr[10])
    sirens_arr_fourth.append(sirens_medium_deep_accuracy_arr[11])
    sirens_arr_fourth.append(sirens_high_deep_accuracy_arr[9])
    sirens_arr_fourth.append(sirens_high_deep_accuracy_arr[10])
    sirens_arr_fourth.append(sirens_high_deep_accuracy_arr[11])

    # data visualization
    # #labels for average graph
    labels_ave = ('Cafeteria', 'People Talking', 'Sirens')
    # labels for three sub graphs
    labels = ('Clean (Old Mozilla)', 'Clean (New Mozilla)', 'Clean (New Scorer)', 'Low Noise (Old Mozilla)', 'Low Noise (New Mozilla)', 'Low Noise (New Scorer)',
              'Medium Noise (Old Mozilla)', 'Medium Noise (New Mozilla)', 'Medium Noise (New Scorer)', 'High Noise (Old Mozilla)', 'High Noise (New Mozilla)', 'High Noise (New Scorer)')

    labels_compare = ('Clean', 'Low Noise', 'Medium Noise', 'High Noise')

    # average accuracy under different noise arrays
    # want to look at the average for the old and new versions

    # variable for number of softwares being tested
    soft = 3

    clean_ave_old = 0
    clean_ave_new = 0
    clean_ave_improved = 0
    for val in range(len(clean_deep_accuracy_arr)):
        if (val % soft) == 0:
            clean_ave_old += clean_deep_accuracy_arr[val]
        elif (val % soft) == 1:
            clean_ave_new += clean_deep_accuracy_arr[val]
        else:
            clean_ave_improved += clean_deep_accuracy_arr[val]
    clean_ave_old = clean_ave_old/(len(clean_deep_accuracy_arr)/soft)
    clean_ave_new = clean_ave_new/(len(clean_deep_accuracy_arr)/soft)
    clean_ave_improved = clean_ave_improved/(len(clean_deep_accuracy_arr)/soft)

    cafe_low_ave_old = 0
    cafe_low_ave_new = 0
    cafe_low_ave_improved = 0
    for val in range(len(cafe_low_deep_accuracy_arr)):
        if (val % soft) == 0:
            cafe_low_ave_old += cafe_low_deep_accuracy_arr[val]
        elif (val %soft) == 1:
            cafe_low_ave_new += cafe_low_deep_accuracy_arr[val]
        else:
            cafe_low_ave_improved += cafe_low_deep_accuracy_arr[val]
    cafe_low_ave_old = cafe_low_ave_old/(len(cafe_low_deep_accuracy_arr)/soft)
    cafe_low_ave_new = cafe_low_ave_new/(len(cafe_low_deep_accuracy_arr)/soft)
    cafe_low_ave_improved = cafe_low_ave_improved/(len(cafe_low_deep_accuracy_arr)/soft)

    people_low_ave_old = 0
    people_low_ave_new = 0
    people_low_ave_improved = 0
    for val in range(len(people_low_deep_accuracy_arr)):
        if (val % soft) == 0:
            people_low_ave_old += people_low_deep_accuracy_arr[val]
        elif (val % soft) == 1:
            people_low_ave_new += people_low_deep_accuracy_arr[val]
        else:
            people_low_ave_improved += val
    people_low_ave_old = people_low_ave_old / \
        (len(people_low_deep_accuracy_arr)/soft)
    people_low_ave_new = people_low_ave_new / \
        (len(people_low_deep_accuracy_arr)/soft)
    people_low_ave_improved = people_low_ave_improved / \
        (len(people_low_deep_accuracy_arr)/soft)

    sirens_low_ave_old = 0
    sirens_low_ave_new = 0
    sirens_low_ave_improved = 0
    for val in range(len(sirens_low_deep_accuracy_arr)):
        if (val % soft) == 0:
            sirens_low_ave_old += sirens_low_deep_accuracy_arr[val]
        elif (val % soft) == 1:
            sirens_low_ave_new += sirens_low_deep_accuracy_arr[val]
        else:
            sirens_low_ave_improved += sirens_low_deep_accuracy_arr[val]
    sirens_low_ave_old = sirens_low_ave_old / \
        (len(sirens_low_deep_accuracy_arr)/soft)
    sirens_low_ave_new = sirens_low_ave_new / \
        (len(sirens_low_deep_accuracy_arr)/soft)
    sirens_low_ave_improved = sirens_low_ave_improved / \
        (len(sirens_low_deep_accuracy_arr)/soft)

    cafe_medium_ave_old = 0
    cafe_medium_ave_new = 0
    cafe_medium_ave_improved = 0
    for val in range(len(cafe_medium_deep_accuracy_arr)):
        if (val % soft) == 0:
            cafe_medium_ave_old += cafe_medium_deep_accuracy_arr[val]
        elif (val % soft) == 1:
            cafe_medium_ave_new += cafe_medium_deep_accuracy_arr[val]
        else:
            cafe_medium_ave_improved += cafe_medium_deep_accuracy_arr[val]
    cafe_medium_ave_old = cafe_medium_ave_old / \
        (len(cafe_medium_deep_accuracy_arr)/soft)
    cafe_medium_ave_new = cafe_medium_ave_new / \
        (len(cafe_medium_deep_accuracy_arr)/soft)
    cafe_medium_ave_improved = cafe_medium_ave_improved / \
        (len(cafe_medium_deep_accuracy_arr)/soft)

    people_medium_ave_old = 0
    people_medium_ave_new = 0
    people_medium_ave_improved = 0
    for val in range(len(people_medium_deep_accuracy_arr)):
        if (val % soft) == 0:
            people_medium_ave_old += people_medium_deep_accuracy_arr[val]
        elif (val % soft) == 1:
            people_medium_ave_new += people_medium_deep_accuracy_arr[val]
        else:
            people_medium_ave_improved += people_medium_deep_accuracy_arr[val]
    people_medium_ave_old = people_medium_ave_old / \
        (len(people_medium_deep_accuracy_arr)/soft)
    people_medium_ave_new = people_medium_ave_new / \
        (len(people_medium_deep_accuracy_arr)/soft)
    people_medium_ave_improved = people_medium_ave_improved / \
        (len(people_medium_deep_accuracy_arr)/soft)

    sirens_medium_ave_old = 0
    sirens_medium_ave_new = 0
    sirens_medium_ave_improved = 0
    for val in range(len(sirens_medium_deep_accuracy_arr)):
        if (val % soft) == 0:
            sirens_medium_ave_old += sirens_medium_deep_accuracy_arr[val]
        elif (val % soft) == 1:
            sirens_medium_ave_new += sirens_medium_deep_accuracy_arr[val]
        else:
            sirens_medium_ave_improved += sirens_medium_deep_accuracy_arr[val]
    sirens_medium_ave_old = sirens_medium_ave_old / \
        (len(sirens_medium_deep_accuracy_arr)/soft)
    sirens_medium_ave_new = sirens_medium_ave_new / \
        (len(sirens_medium_deep_accuracy_arr)/soft)
    sirens_medium_ave_improved = sirens_medium_ave_improved / \
        (len(sirens_medium_deep_accuracy_arr)/soft)

    cafe_high_ave_old = 0
    cafe_high_ave_new = 0
    cafe_high_ave_improved = 0
    for val in range(len(cafe_high_deep_accuracy_arr)):
        if (val % soft) == 0:
            cafe_high_ave_old += cafe_high_deep_accuracy_arr[val]
        elif (val % soft) == 1:
            cafe_high_ave_new += cafe_high_deep_accuracy_arr[val]
        else:
            cafe_high_ave_improved += cafe_high_deep_accuracy_arr[val]
    cafe_high_ave_old = cafe_high_ave_old/(len(cafe_high_deep_accuracy_arr)/soft)
    cafe_high_ave_new = cafe_high_ave_new/(len(cafe_high_deep_accuracy_arr)/soft)
    cafe_high_ave_improved = cafe_high_ave_improved/(len(cafe_high_deep_accuracy_arr)/soft)

    people_high_ave_old = 0
    people_high_ave_new = 0
    people_high_ave_improved = 0
    for val in range(len(people_high_deep_accuracy_arr)):
        if (val % soft) == 0:
            people_high_ave_old += people_high_deep_accuracy_arr[val]
        elif (val % soft) == 1:
            people_high_ave_new += people_high_deep_accuracy_arr[val]
        else:
            people_high_ave_improved += people_high_deep_accuracy_arr[val]
    people_high_ave_old = people_high_ave_old / \
        (len(people_high_deep_accuracy_arr)/soft)
    people_high_ave_new = people_high_ave_new / \
        (len(people_high_deep_accuracy_arr)/soft)
    people_high_ave_improved = people_high_ave_improved / \
        (len(people_high_deep_accuracy_arr)/soft)

    sirens_high_ave_old = 0
    sirens_high_ave_new = 0
    sirens_high_ave_improved = 0
    for val in range(len(sirens_high_deep_accuracy_arr)):
        if (val % soft) == 0:
            sirens_high_ave_old += sirens_high_deep_accuracy_arr[val]
        elif (val % soft) == 1:
            sirens_high_ave_new += sirens_high_deep_accuracy_arr[val]
        else:
            sirens_high_ave_improved += sirens_high_deep_accuracy_arr[val]
    sirens_high_ave_old = sirens_high_ave_old / \
        (len(sirens_high_deep_accuracy_arr)/soft)
    sirens_high_ave_new = sirens_high_ave_new / \
        (len(sirens_high_deep_accuracy_arr)/soft)
    sirens_high_ave_improved = sirens_high_ave_improved / \
        (len(sirens_high_deep_accuracy_arr)/soft)

    clean_arr_old = []
    clean_arr_old.append(clean_ave_old)
    clean_arr_old.append(clean_ave_old)
    clean_arr_old.append(clean_ave_old)

    clean_arr_new = []
    clean_arr_new.append(clean_ave_new)
    clean_arr_new.append(clean_ave_new)
    clean_arr_new.append(clean_ave_new)

    clean_arr_improved = []
    clean_arr_improved.append(clean_ave_improved)
    clean_arr_improved.append(clean_ave_improved)
    clean_arr_improved.append(clean_ave_improved)

    low_arr_old = []
    low_arr_old.append(cafe_low_ave_old)
    low_arr_old.append(people_low_ave_old)
    low_arr_old.append(sirens_low_ave_old)

    low_arr_new = []
    low_arr_new.append(cafe_low_ave_new)
    low_arr_new.append(people_low_ave_new)
    low_arr_new.append(sirens_low_ave_new)

    low_arr_improved = []
    low_arr_improved.append(cafe_low_ave_improved)
    low_arr_improved.append(people_low_ave_improved)
    low_arr_improved.append(sirens_low_ave_improved)

    medium_arr_old = []
    medium_arr_old.append(cafe_medium_ave_old)
    medium_arr_old.append(people_medium_ave_old)
    medium_arr_old.append(sirens_medium_ave_old)

    medium_arr_new = []
    medium_arr_new.append(cafe_medium_ave_new)
    medium_arr_new.append(people_medium_ave_new)
    medium_arr_new.append(sirens_medium_ave_new)

    medium_arr_improved = []
    medium_arr_improved.append(cafe_medium_ave_improved)
    medium_arr_improved.append(people_medium_ave_improved)
    medium_arr_improved.append(sirens_medium_ave_improved)

    high_arr_old = []
    high_arr_old.append(cafe_high_ave_old)
    high_arr_old.append(people_high_ave_old)
    high_arr_old.append(sirens_high_ave_old)

    high_arr_new = []
    high_arr_new.append(cafe_high_ave_new)
    high_arr_new.append(people_high_ave_new)
    high_arr_new.append(sirens_high_ave_new)

    high_arr_improved = []
    high_arr_improved.append(cafe_high_ave_improved)
    high_arr_improved.append(people_high_ave_improved)
    high_arr_improved.append(sirens_high_ave_improved)

    # Comparing the different version of DeepSpeech

    deep_speech_per_old = []
    deep_speech_per_new = []
    deep_speech_per_improved = []

    # low ave for both deepspeech versions
    deep_low_sum_old = cafe_low_ave_old + people_low_ave_old + sirens_low_ave_old
    deep_low_sum_new = cafe_low_ave_new + people_low_ave_new + sirens_low_ave_new
    deep_low_sum_improved = cafe_low_ave_improved + people_low_ave_improved + sirens_low_ave_improved

    deep_low_ave_old = deep_low_sum_old / 3
    deep_low_ave_new = deep_low_sum_new / 3
    deep_low_ave_improved = deep_low_sum_improved / 3

    # medium ave for both google and deep
    deep_medium_sum_old = cafe_medium_ave_old + \
        people_medium_ave_old + sirens_medium_ave_old
    deep_medium_sum_new = cafe_medium_ave_new + \
        people_medium_ave_new + sirens_medium_ave_new
    deep_medium_sum_improved = cafe_medium_ave_improved + people_medium_ave_improved + sirens_medium_ave_improved

    deep_medium_ave_old = deep_medium_sum_old / 3
    deep_medium_ave_new = deep_medium_sum_new / 3
    deep_medium_ave_improved = deep_medium_sum_improved / 3

    # high ave for both google and deep
    deep_high_sum_old = cafe_high_ave_old + \
        people_high_ave_old + sirens_high_ave_old
    deep_high_sum_new = cafe_high_ave_new + \
        people_high_ave_new + sirens_high_ave_new
    deep_high_sum_improved = cafe_high_ave_improved + people_high_ave_improved + sirens_high_ave_improved

    deep_high_ave_old = deep_high_sum_old / 3
    deep_high_ave_new = deep_high_sum_new / 3
    deep_high_ave_improved = deep_high_sum_improved/ 3

    deep_speech_per_old.append(clean_ave_old)
    deep_speech_per_old.append(deep_low_ave_old)
    deep_speech_per_old.append(deep_medium_ave_old)
    deep_speech_per_old.append(deep_high_ave_old)

    deep_speech_per_new.append(clean_ave_new)
    deep_speech_per_new.append(deep_low_ave_new)
    deep_speech_per_new.append(deep_medium_ave_new)
    deep_speech_per_new.append(deep_high_ave_new)

    deep_speech_per_improved.append(clean_ave_improved)
    deep_speech_per_improved.append(deep_low_ave_improved)
    deep_speech_per_improved.append(deep_medium_ave_improved)
    deep_speech_per_improved.append(deep_high_ave_improved)

    # visualization for cafeteria
    df = pd.DataFrame(np.c_[cafe_arr_first, cafe_arr_second, cafe_arr_third, cafe_arr_fourth],
                      index=labels, columns=['Paramedic Smith', 'EMT 107', 'EMT 117', 'EMT 101'])

    ax = df.plot.bar()
    # ax.set_xlabel("audio file")
    ax.set_ylabel("Accuracy")
    plt.xticks(rotation=360)
    plt.suptitle("Cafeteria Noise Figure")
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig('cafe.png', dpi=100)

    # #visualization for people noise
    df = pd.DataFrame(np.c_[people_arr_first, people_arr_second, people_arr_third, people_arr_fourth],
                      index=labels, columns=['Paramedic Smith', 'EMT 107', 'EMT 117', 'EMT 101'])

    ax = df.plot.bar()
    # ax.set_xlabel("audio file")
    ax.set_ylabel("Accuracy")
    plt.xticks(rotation=360)
    plt.suptitle("People Talking")
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig('ppl.png', dpi=100)

    # visualication for Sirens Noise
    df = pd.DataFrame(np.c_[sirens_arr_first, sirens_arr_second, sirens_arr_third, sirens_arr_fourth],
                      index=labels, columns=['Paramedic Smith', 'EMT 107', 'EMT 117', 'EMT 101'])

    ax = df.plot.bar()
    # ax.set_xlabel("audio file")
    ax.set_ylabel("Accuracy")
    plt.xticks(rotation=360)
    plt.suptitle("Sirens Noise")
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig('siren.png', dpi=100)

    # visualiczation for Average accuracy under dfferent noise profiles
    df = pd.DataFrame(np.c_[clean_arr_old, low_arr_old, medium_arr_old, high_arr_old],
                      index=labels_ave, columns=['Clean', 'Low Noise', 'Medium Noise', 'High Noise'])
    ax = df.plot.bar()
    # ax.set_xlabel("audio file")
    ax.set_ylabel("Accuracy")
    plt.xticks(rotation=360)
    plt.suptitle(
        "Mozilla Deepspeech Recognition v0.6 Performance under different noise profiles")
    fig = plt.gcf()
    fig.savefig('deep_ave.png', dpi=100)

    # visualiczation for Average accuracy under dfferent noise profiles
    df = pd.DataFrame(np.c_[clean_arr_new, low_arr_new, medium_arr_new, high_arr_new],
                      index=labels_ave, columns=['Clean', 'Low Noise', 'Medium Noise', 'High Noise'])
    ax = df.plot.bar()
    # ax.set_xlabel("audio file")
    ax.set_ylabel("Accuracy")
    plt.xticks(rotation=360)
    plt.suptitle(
        "Mozilla Deepspeech Recognition v0.9 Performance under different noise profiles")
    fig = plt.gcf()
    fig.savefig('deep_ave.png', dpi=100)

    # visualization for Average performance for google speech and deepspeech
    df = pd.DataFrame(np.c_[deep_speech_per_old, deep_speech_per_new],
                      index=labels_compare, columns=['V0.6', 'v0.9'])
    ax = df.plot.bar()
    # ax.set_xlabel("audio file")
    ax.set_ylabel("Accuracy")
    plt.xticks(rotation=360)
    plt.suptitle("Older DeepSpeech Recognition VS. new Deepspeech Recognition")
    fig = plt.gcf()
    fig.savefig('compare.png', dpi=100)
