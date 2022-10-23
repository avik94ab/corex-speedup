import pandas as pd
from itertools import product
import math
import numpy as np
'''
THERMODYNAMIC PARAMETERS INITIALIZATION
'''
aCp=0.44
bCp=-0.26
adeltaH=-8.44
bdeltaH=31.4
OHCp=0.17
TsPolar=335.15
TsApolar=385.15
dSbb_length_corr=-0.12
exposed_criteria=0.1
W_Sconf=1.0
Current_Temp = 298.15
ASA_exposed_amide=7.5


def partition_generator(seq_length, window_size, Minimum_Window_Size):
    startIndex = list(range(window_size))
    full_windows = seq_length // window_size
    partitionSchemes = {}
    partitionId = 1
    for s in startIndex:
        windows = []
        start = s
        for i in range(full_windows):
            if (start + window_size - 1 > seq_length - 1):
                windows.append([start, seq_length - 1])
            else:
                windows.append([start, start + window_size - 1])
            start = start + window_size
        if windows[-1][1] < seq_length - 1:
            windows.append([windows[-1][1] + 1, seq_length - 1])
        if (windows[0][0] - 0 <= window_size and windows[0][0 != 0]):
            windows.insert(0, [0, windows[0][0] - 1])

        if(windows[-1][1] - windows[-1][0] + 1 < Minimum_Window_Size):
            windows.remove(windows[-1])
            windows[-1][1] = seq_length - 1

        if(windows[0][1] - windows[0][0] + 1 < Minimum_Window_Size):
            windows.remove(windows[0])
            windows[0][0] = 0

        for w in windows:
            w[0] += 1
            w[1] += 1

        partitionSchemes[partitionId] = windows
        partitionId += 1


    numEnsembles = 0
    for partitionId, partition in partitionSchemes.items():
        print("Partition Scheme No.", partitionId, " contains ", 2** len(partition) - 2, " intermediate states")
        unitId = 1
        for segment in partition:
            print("Unit no.", unitId, "Starting residue:", segment[0], " Final residue:", segment[1])
            unitId += 1

        print("-------")
        numEnsembles += 2 ** len(partition) - 2
    print("Total number of ensembles: ", numEnsembles + 2) #no. of ensemble states +U+F
    print("Total number of partially folded states: ", numEnsembles)
    return partitionSchemes


def readPDB(fileName):
    f = open(fileName, "r")
    lines = f.readlines()
    fileSize = int(lines[0].strip()) #no. of rows in the file
    headers = lines[1].split(' ')
    lines = lines[2:]
    for i in range(len(headers)):
        headers[i] = headers[i].strip("\n")

    df = pd.DataFrame(columns=headers, index=list(range(fileSize)))

    index = 0
    for line in lines:
        words = line.split(" ")
        lst = []
        for word in words:
            if word != "" and word != "\n":
                lst.append(word)
        df.iloc[index] = lst[:-1]
        index += 1

    seq_length = int(df.loc[fileSize-1]['ResNum'])
    print("Successfully read .pdb file of sequence length: ", seq_length, "\n")
    print("Total no.of rows in the file: ", fileSize, "\n")

    return fileSize, seq_length, df


def state_generator(partitionSchemes):
    partitionStates = {}
    for partitionId, partition in partitionSchemes.items():
        tmp = []
        for item in product([0,1], repeat=len(partition)):
            if sum(item)!=0 and sum(item)<len(partition):
                tmp.append(list(item))
        partitionStates[partitionId] = tmp
    return partitionStates


def getAAConstants():
    aaConstants = pd.read_csv("aaConstants.csv")
    aaConstants = aaConstants.iloc[:, 1:]
    aaConstants = aaConstants.set_index('aa')
    return aaConstants


def Native_State(partitionId, partitionSchemes, partitionStates, df):

    aaConstants = getAAConstants()
    ASA_N_Apolar = 0.0
    ASA_N_Polar = 0.0
    ASA_N_Apolar_unit = [0.0] * len(partitionSchemes[partitionId])
    ASA_N_Polar_unit = [0.0] * len(partitionSchemes[partitionId])
    ASA_U_Apolar_unit = [0.0] * len(partitionSchemes[partitionId])
    ASA_U_Polar_unit = [0.0] * len(partitionSchemes[partitionId])

    for i in range(len(partitionSchemes[partitionId])):
        for j in range(partitionSchemes[partitionId][i][0], partitionSchemes[partitionId][i][1] + 1):
            ASA_side_chain = 0.0
            print(j, end= ":")
            k = df.index[df['ResNum'] == str(j)].tolist()
            print(k)
            for atom in k:
                element = df.at[atom, 'AtomName']
                if element == 'C':
                    ASA_N_Apolar_unit[i] += float(df.at[atom, 'Nat.Area'])
                    ASA_N_Apolar = ASA_N_Apolar + float(df.at[atom, 'Nat.Area'])
                else:
                    ASA_N_Polar_unit[i] += float(df.at[atom, 'Nat.Area'])
                    ASA_N_Polar = ASA_N_Polar + float(df.at[atom, 'Nat.Area'])
                if element not in ["N", "CA", "C", "O"]:
                    ASA_side_chain = ASA_side_chain + float(df.at[atom, 'Nat.Area'])

            idx = df.index[df['ResNum'] == str(j)].tolist()[0]
            aminoAcid = df.at[idx, 'ResName']
            ASA_U_Apolar_unit[i] = ASA_U_Apolar_unit[i] + aaConstants.at[aminoAcid, 'ASAexapol']
            ASA_U_Polar_unit[i] = ASA_U_Polar_unit[i] + aaConstants.at[aminoAcid, 'ASAexpol']

        print('------')

        #return two areas calculated for the partitionID and ASA-side-chain



fileSize, seq_length, df = readPDB("1ediA.pdb.info")

partitionSchemes = partition_generator(seq_length, 5, 4)

partitionStates = state_generator(partitionSchemes)

Native_State(1, partitionSchemes, partitionStates, df)

#print(df[['ResName', 'ResNum', 'AtomNum', 'AtomName', 'Nat.Area']])

#print("Length of the sequence is: ", seq_length)

print(df)

output = ""
for k,v in partitionStates.items():
    num = 1
    for f in v:
        output += str(k) + "  "+ str(num) + "  "+str(num) + "  " + str(round(np.sum(f)/len(f),4)) + "\n"
        num += 1

text_file = open("data.txt", "w")
text_file.write(output)
text_file.close()