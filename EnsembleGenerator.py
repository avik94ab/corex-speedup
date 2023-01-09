# 0: Folded, 1: Unfolded
#num_atoms is the no. of folded atoms
import itertools
import time, sys
import random
#for timing calculations

#general purpose imports
import pandas as pd
from itertools import product
import math
import numpy as np
from biotite.structure import sasa
import biotite.structure.io as strucio
import infoStable
from numba import jit, njit

#multiprocessing library
from multiprocessing import Pool

import pdbIO, SFR

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
W_Sconf=0.5 # might be an user input later
Current_Temp = 298.15
ASA_exposed_amide=7.5


'''
Radius of atoms table
'''
radius_table = {"N": 1.650, "CA": 1.870, "C": 1.760,
                "O": 1.400, "AD1": 1.700, "AD2": 1.700,
                "AE1": 1.700, "AE2": 1.700, "CB": 1.870,
                "CD": 1.870, "CD1": 1.760, "CD2": 1.760,
                "CG": 1.870, "CG1": 1.870, "CG2": 1.870,
                "CE": 1.870, "CE1": 1.760, "CE2": 1.760,
                "CE3": 1.760, "CH2": 1.760, "CZ": 1.870,
                "CZ2": 1.760, "CZ3": 1.760, "ND": 1.650,
                "ND1": 1.650, "ND2": 1.650, "NE": 1.650,
                "NE1": 1.650, "NE2": 1.650, "NH1": 1.650,
                "NH2": 1.650, "NZ": 1.650, "OD": 1.400,
                "OD1": 1.400, "OD2": 1.400, "OT2": 1.400,
                "OE": 1.400, "OE1": 1.400, "OE2": 1.400,
                "OH2": 1.400, "OG": 1.400, "OG1": 1.400,
                "OT1": 1.400, "OXT": 1.400, "OH": 1.400,
                "SD": 1.850, "SG": 1.850, "P": 1.900
                }



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
    print("Total number of micro-states: ", numEnsembles + 2) #no. of ensemble states +U+F
    print("Total number of partially folded micro-states: ", numEnsembles)
    return partitionSchemes

def readPDBfile(fileName):
    '''
    atom_lst = []
    f = open(fileName, "r")
    lines = f.readlines()
    OTnum = 0
    for line in lines:
        if line[0:4] == "ATOM":
            atom_lst.append([line.split()[3], line.split()[5], line.split()[2],[line.split()[6], line.split()[7], line.split()[8]],
                             radius_table[line.split()[2]], math.pi * (radius_table[line.split()[2]]**2)])
            if line.split()[2] == "OXT" or line.split()[2] == "OT":
                OTnum += 1
    '''

    OTnum = 0

    stack_from_pdb = strucio.load_structure(fileName)
    atom_lst = []
    vdw_radii = []
    for idx in range(len(stack_from_pdb)):
        vdw_radii.append(radius_table[stack_from_pdb.get_atom(idx).atom_name])
    vdw_radii = np.array(vdw_radii)
    atom_sasa_exp = sasa(stack_from_pdb, point_number=1000, vdw_radii=vdw_radii)

    for idx in range(len(stack_from_pdb)):
        atom_lst.append((stack_from_pdb.get_atom(idx).res_name, stack_from_pdb.get_atom(idx).atom_name, stack_from_pdb.get_atom(idx).coord, vdw_radii[idx], atom_sasa_exp[idx]))
        if stack_from_pdb.get_atom(idx).atom_name == "OXT" or stack_from_pdb.get_atom(idx).atom_name == "OT":
            OTnum += 1

    return OTnum, atom_lst

def readPDBinfo(fileName):
    f = open(fileName, "r")
    lines = f.readlines()
    fileSize = int(lines[0].strip())  #no. of rows in the file
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
        if partitionId == len(partitionSchemes):
            tmp.append(list([1]*len(item)))
        partitionStates[partitionId] = tmp
    return partitionStates


def getAAConstants():
    aaConstants = pd.read_csv("aaConstants.csv")
    aaConstants = aaConstants.iloc[:, 1:]
    aaConstants = aaConstants.set_index('aa')
    return aaConstants


def Native_State(partitionId, partitionSchemes, df, OTnum):

    seq_length = int(df['ResNum'].iloc[-1])
    aaConstants = getAAConstants()
    ASA_N_Apolar = 0.0
    ASA_N_Polar = 0.0

    ASA_N_Apolar_unit = [0.0] * len(partitionSchemes[partitionId])
    ASA_N_Polar_unit = [0.0] * len(partitionSchemes[partitionId])
    ASA_U_Apolar_unit = [0.0] * len(partitionSchemes[partitionId])
    ASA_U_Polar_unit = [0.0] * len(partitionSchemes[partitionId])
    Fraction_exposed_Native = [0.0] * seq_length

    for i in range(len(partitionSchemes[partitionId])):
        for j in range(partitionSchemes[partitionId][i][0], partitionSchemes[partitionId][i][1] + 1):
            ASA_side_chain = 0.0
            #print(j, end = ":")
            k = df.index[df['ResNum'] == j].tolist()
            for atom in k:
                element = df.at[atom, 'AtomName']
                if element[0] == 'C':
                    ASA_N_Apolar_unit[i] += float(df.at[atom, 'Nat.Area'])
                    ASA_N_Apolar = ASA_N_Apolar + float(df.at[atom, 'Nat.Area'])
                else:
                    ASA_N_Polar_unit[i] += float(df.at[atom, 'Nat.Area'])
                    ASA_N_Polar = ASA_N_Polar + float(df.at[atom, 'Nat.Area'])
                if element not in ["N", "CA", "C", "O"]:
                    ASA_side_chain = ASA_side_chain + float(df.at[atom, 'Nat.Area'])
            idx = df.index[df['ResNum'] == j].tolist()[0]
            aminoAcid = df.at[idx, 'ResName']
            ASA_U_Apolar_unit[i] = ASA_U_Apolar_unit[i] + aaConstants.at[aminoAcid, 'ASAexapol']
            ASA_U_Polar_unit[i] = ASA_U_Polar_unit[i] + aaConstants.at[aminoAcid, 'ASAexpol']
            Fraction_exposed_Native[j - 1] = ASA_side_chain / aaConstants.at[aminoAcid, 'ASAsc']

        #print('------')

    ASA_U_Polar_unit[0] = ASA_U_Polar_unit[0] + 45.0
    j = len(partitionSchemes[partitionId]) - 1
    ASA_U_Apolar_unit[j] = ASA_U_Apolar_unit[j] + 30.0
    ASA_U_Polar_unit[j] = ASA_U_Polar_unit[j] + 30.0 * OTnum



    return ASA_N_Apolar, ASA_N_Polar, ASA_N_Apolar_unit, ASA_N_Polar_unit, ASA_U_Apolar_unit, ASA_U_Polar_unit, Fraction_exposed_Native




def load_atoms_range(partitionId, partitionSchemes, partitionStates, stateNum, df):
    '''
    @param partitionId
    @param partitionSchemes
    @param partitionStates
    @param stateNum
    @param df
    '''
    state = partitionStates[partitionId][stateNum]
    num_atoms = 0
    num_residues = 0
    folded_atoms = []
    for i in range(len(partitionSchemes[partitionId])):
        if state[i] == 0:
            for j in range(partitionSchemes[partitionId][i][0], partitionSchemes[partitionId][i][1] + 1):
                num_residues += 1
                k = df.index[df['ResNum'] == j].tolist()
                num_atoms += len(k)
                folded_atoms.append(k)

    folded_atoms = list(itertools.chain(*folded_atoms))


    return state, num_atoms, num_residues, folded_atoms


def calc_ASA_dSconf(folded_atoms, partitionId, partitionStates, partitionSchemes, stateNum, ASA_U_Apolar_unit, ASA_U_Polar_unit, ASA_N_Apolar, ASA_N_Polar,Fraction_exposed_Native, df, stack_from_pdb):

    seq_length = int(df['ResNum'].iloc[-1])
    aaConstants =getAAConstants()
    ASA_State_Apolar = 0.0
    ASA_State_Polar = 0.0
    Sum_U_Apolar = 0.0
    Sum_U_Polar = 0.0

    Fraction_exposed = [0.0] * seq_length
    Fraction_amide_exposed = [0.0] * seq_length

    #_, _, pdb_df, _, stack_from_pdb = pdbIO.readPDB('6cne')


    area_folded = pdbIO.asa_calc(stack_from_pdb, folded_atoms)
    '''
    area_folded = {0: 20.816296, 1: 30.736069, 2: 0.0, 3: 0.0, 4: 4.355689, 5: 0.0, 6: 1.699377, 7: 5.872103, 8: 22.668797,
     9: 25.548208, 10: 13.910856, 11: 21.221872, 12: 24.160765, 13: 0.0, 14: 0.0, 15: 0.19223, 16: 0.0, 17: 0.553688,
     18: 0.0, 19: 7.189323, 20: 0.0, 21: 0.53595, 22: 0.0, 23: 0.0, 24: 13.841323, 25: 0.0, 26: 0.0, 27: 0.628103,
     28: 15.799906, 29: 12.600565, 30: 0.584385, 31: 15.434875, 32: 20.598974, 33: 43.051632, 34: 0.0, 35: 5.705573,
     36: 0.0, 37: 0.0, 38: 0.25958, 39: 10.434772, 40: 18.912025, 41: 16.109144, 42: 5.26179, 43: 0.0, 44: 0.0,
     45: 19.789999, 46: 6.474614, 47: 4.155155, 48: 32.739361, 49: 49.403614, 50: 0.0, 51: 2.158533, 52: 0.0,
     53: 3.72902, 54: 0.0, 55: 0.0, 56: 13.353753, 57: 11.642921, 58: 7.660136, 59: 0.0, 60: 2.6337, 61: 24.903717,
     62: 15.563202, 63: 12.798459, 64: 6.640003, 65: 56.513687, 66: 0.0, 67: 10.836683, 68: 2.717375, 69: 0.0,
     70: 4.698823, 71: 10.356906, 72: 0.0, 73: 26.398033, 74: 14.762012, 75: 23.1514, 76: 39.152264, 77: 39.920197,
     78: 49.597851, 79: 1.587399, 80: 5.73213, 81: 3.283231, 82: 31.213619, 83: 14.510485, 84: 28.877146, 85: 56.012684,
     86: 0.411099, 87: 1.166114, 88: 0.0, 89: 0.0, 90: 8.068741, 91: 0.0, 92: 44.950436, 93: 8.582023, 94: 6.481269,
     95: 0.0, 96: 0.0, 97: 24.475214, 98: 17.868851, 99: 2.556142, 100: 35.656116, 101: 36.371655, 102: 17.50424,
     103: 0.0, 104: 31.163214, 105: 0.0, 106: 0.314338, 107: 5.994563, 108: 0.0, 109: 0.732711, 110: 26.8895,
     111: 8.033434, 112: 9.770435, 113: 14.647825, 114: 27.205275, 115: 28.316706, 116: 0.0, 117: 7.42271, 118: 0.0,
     119: 0.0, 120: 0.0, 121: 11.168184, 122: 42.517319, 123: 1.47309, 124: 0.0, 125: 0.0, 126: 15.882969, 127: 0.0,
     128: 27.617132, 129: 35.37542, 130: 0.0, 131: 6.742195, 132: 0.0, 133: 0.0, 134: 5.02066, 135: 20.485931, 136: 0.0,
     137: 7.069132, 138: 0.0, 139: 0.0, 140: 19.474607, 141: 21.095272, 142: 29.514874, 143: 14.554958, 144: 16.573122,
     145: 4.112822, 146: 0.0, 147: 6.258868, 148: 1.123382, 149: 0.0, 150: 4.156476, 151: 1.056901, 152: 9.664666,
     153: 0.0, 154: 26.753269, 155: 11.576353, 156: 65.044998, 157: 27.100046, 158: 0.0, 159: 1.076416, 160: 0.0,
     161: 0.0, 162: 29.077278, 163: 0.687763, 164: 0.0, 165: 34.819275, 166: 3.170624, 167: 0.816384, 168: 3.021756,
     169: 2.682714, 170: 46.600246, 171: 1.660009, 172: 3.734929, 173: 0.0, 174: 0.0, 175: 48.692631, 176: 0.0,
     177: 0.0, 178: 0.0, 179: 0.0, 180: 0.0, 181: 14.040751, 182: 33.348511, 183: 0.0, 184: 0.0, 185: 0.0, 186: 0.0,
     187: 3.74059, 188: 0.87942, 189: 6.535676, 190: 0.0, 191: 9.674044, 192: 10.365067, 193: 22.880775, 194: 24.025543,
     195: 23.669575, 196: 36.116512, 197: 0.0, 198: 6.755713, 199: 0.734675, 200: 25.740376, 201: 13.922054,
     202: 13.964081, 203: 18.23893, 204: 27.751387, 205: 54.555679, 206: 0.0, 207: 8.73107, 208: 0.0, 209: 20.402407,
     210: 0.0, 211: 33.319489, 212: 38.30986, 213: 0.0, 214: 1.425958, 215: 16.692577, 216: 27.265825, 217: 8.776429,
     218: 0.0, 219: 4.74687, 220: 11.08367, 221: 0.398154, 222: 3.233413, 223: 0.0, 267: 38.37011, 268: 11.159319,
     269: 0.537967, 270: 28.807606, 271: 16.813705, 272: 22.940609, 273: 26.992781, 274: 34.743599, 275: 2.587436,
     276: 5.381086, 277: 5.321254, 278: 18.575571, 279: 8.065374, 280: 8.281933, 281: 38.952164, 282: 39.268986,
     283: 4.968644, 284: 43.878231, 285: 3.753955, 286: 27.983643, 287: 2.443358, 288: 3.413313, 289: 0.0,
     290: 28.653788, 291: 9.901397, 292: 32.218006, 293: 6.153358, 294: 5.189286, 295: 7.939446, 296: 29.812971,
     297: 36.617115, 298: 25.253223, 299: 21.210253, 300: 35.984329, 301: 29.426035, 348: 54.210499, 349: 9.94329,
     350: 4.097239, 351: 1.205872, 352: 27.407187, 353: 11.948113, 354: 6.588119, 355: 27.893965, 356: 8.049112,
     357: 3.309474, 358: 0.768666, 359: 3.367748, 360: 29.208694, 361: 15.596857, 362: 10.20172, 363: 34.774345,
     364: 3.560763, 365: 11.553741, 366: 0.0, 367: 27.141317, 368: 50.691525, 369: 0.0, 370: 7.135497, 371: 2.178325,
     372: 17.907433, 373: 14.924515, 374: 11.201432, 375: 46.151688, 376: 0.0, 377: 3.731828, 378: 17.225246, 379: 0.0,
     380: 2.118421, 381: 12.965604, 382: 0.0, 383: 22.398594, 384: 36.745739}
     '''
    for i in folded_atoms:
        element = df.at[i, 'AtomName']
        if element[0] == "C":
            ASA_State_Apolar = ASA_State_Apolar + area_folded[i]
        else:
            ASA_State_Polar = ASA_State_Polar + area_folded[i]

    Sconf = 0.0
    state = partitionStates[partitionId][stateNum]

    for i in range(len(partitionSchemes[partitionId])):
        if state[i] == 1:
            Sum_U_Apolar = Sum_U_Apolar + ASA_U_Apolar_unit[i]
            Sum_U_Polar = Sum_U_Polar + ASA_U_Polar_unit[i]
            for j in range(partitionSchemes[partitionId][i][0], partitionSchemes[partitionId][i][1] + 1):
                idx = df.index[df['ResNum'] == j].tolist()[0]
                aminoAcid = df.at[idx, 'ResName']
                Sconf = Sconf + aaConstants.at[aminoAcid, 'dSexu'] + aaConstants.at[aminoAcid, 'dSbb'] + dSbb_length_corr + (1.0 - Fraction_exposed_Native[j-1]) * aaConstants.at[aminoAcid, 'dSbuex']


    for i in range(len(partitionSchemes[partitionId])):
        if state[i] == 0:
            for j in range(partitionSchemes[partitionId][i][0],partitionSchemes[partitionId][i][1]+1):
                ASA_side_chain = 0.0
                ASA_Amide = 0.0
                k = df.index[df['ResNum'] == j].tolist()
                for atom in k:
                    element = df.at[atom, 'AtomName']
                    if element[0] == "N": #check with Jamie
                        ASA_Amide = ASA_Amide + area_folded[atom]
                    if element not in ["N", "CA", "C", "O"]: #check with Jamie
                        ASA_side_chain = ASA_side_chain + area_folded[atom]
                idx = df.index[df['ResNum'] == j].tolist()[0]
                aminoAcid = df.at[idx, 'ResName']
                Fraction_exposed[j-1] = ASA_side_chain / aaConstants.at[aminoAcid, 'ASAsc']
                Fraction_amide_exposed[j-1] = ASA_Amide / ASA_exposed_amide
                Sconf = Sconf + (Fraction_exposed[j-1] - Fraction_exposed_Native[j-1]) * aaConstants.at[aminoAcid, 'dSbuex']
                #print(j, aminoAcid, ", Sconf = ", Sconf, ", Fraction_exposed[j-1] = ", Fraction_exposed[j-1], ", Fraction_exposed_Native[j-1] = ",Fraction_exposed_Native[j-1],  ", dSbuex = ",aaConstants.at[aminoAcid, 'dSbuex'], "ASA_side_chain = ", ASA_side_chain, "ASA_sc = ", aaConstants.at[aminoAcid, 'ASAsc'])
                #print(j, aminoAcid, ", Sconf = ", Sconf, ", Fraction_exposed[j-1] = ", Fraction_exposed[j-1], ", Fraction_exposed_Native[j-1] = ",Fraction_exposed_Native[j-1],  ", dSbuex = ",aaConstants.at[aminoAcid, 'dSbuex'])


    delASA_ap = ASA_State_Apolar + Sum_U_Apolar - ASA_N_Apolar
    delASA_pol = ASA_State_Polar + Sum_U_Polar - ASA_N_Polar

    return Sconf, delASA_ap, delASA_pol



def task(args):
    partitionId, partitionSchemes, partitionStates, partition , df, stack_from_pdb, OTnum = args
    ASA_N_Apolar, ASA_N_Polar, ASA_N_Apolar_unit, ASA_N_Polar_unit, ASA_U_Apolar_unit, ASA_U_Polar_unit, Fraction_exposed_Native = Native_State(partitionId, partitionSchemes, df, OTnum)
    x = load_atoms_range(partitionId, partitionSchemes, partitionStates, partition, df)
    y = calc_ASA_dSconf(x[3], partitionId, partitionStates, partitionSchemes, partition, ASA_U_Apolar_unit, ASA_U_Polar_unit, ASA_N_Apolar, ASA_N_Polar, Fraction_exposed_Native,df, stack_from_pdb)
    Fraction_folded = partitionStates[partitionId][partition].count(0) / len(partitionStates[partitionId][partition])
    stateFlag = ""
    for f in partitionStates[partitionId][partition]:
        stateFlag += str(f)
    return partitionId, partition, Fraction_folded, y[0], y[1], y[2], stateFlag





if __name__ == '__main__':

    #fileSize, seq_length, df = readPDBinfo("1ediA.pdb.info")

    #seq_length, OTnum, df = infoStable.trial()

    #seq_length, OTnum, pdb_lst, df, stack_from_pdb = pdbIO.readPDB('6cne')

    #print(type(df.at[1, 'ResNum']))

    pdbID = input("Enter pdb id of protein: ")
    window_size = int(input("Enter window size: "))
    Minimum_Window_Size = int(input("Enter minimum window size: "))

    seq_length, OTnum, pdb_lst, df, stack_from_pdb = pdbIO.readPDB(pdbID)

    print(df)

    #print(df.index[df['ResNum'] == 56].tolist()[0])

    #print(type(df.at[1, 'ResNum']))

    partitionSchemes = partition_generator(seq_length, window_size, Minimum_Window_Size)

    partitionStates = state_generator(partitionSchemes)


    #TODO: Insert OTnum from readPDBfile
    output = []



    args = [] #list of all possible arguement combination

    for partitionId, partition in partitionStates.items():
        for i in range(len(partition)):
            args.append((partitionId, partitionSchemes, partitionStates, i , df, stack_from_pdb, OTnum))

    start_time = time.time()
    with Pool(5) as pool:
        result = pool.map(task, args)


    tmpStr = []
    for r in result:
        tmpStr.append(str(r[0]) +  ' '+ str(r[1]+1) + ' ' + str(r[1]+1) + ' ' +str(r[2]) + ' '+ str(r[3]) + ' ' + str(r[4]) + ' ' + str(r[5]) + ' '+ str(r[6]) + '\n')
    output = "".join(tmpStr)

    text_file = open(pdbID+".pdb"+str(window_size)+"."+str(Minimum_Window_Size), "wt")
    n = text_file.write(output)
    text_file.close()

    print("\n--- %s seconds---" % (time.time() - start_time))

    Partition_Function25 = 1.0

    file = open(pdbID+".pdb"+str(window_size)+"."+str(Minimum_Window_Size), 'r')
    ensemble = file.readlines()
    file.close()

    ln_kf = SFR.Residue_Probabilities(ensemble, seq_length, partitionSchemes, partitionStates, Partition_Function25)

    corexOut = []
    for i in range(len(ln_kf)):
        corexOut.append(((i+1), df.at[df.index[df['ResNum'] == (i+1)].tolist()[0], 'ResName'], ln_kf[i]))
        print(i+1, ln_kf[i])






    '''

    for partitionId, partition in partitionStates.items():
        for i in range(len(partition)):
            ASA_N_Apolar, ASA_N_Polar, ASA_N_Apolar_unit, ASA_N_Polar_unit, ASA_U_Apolar_unit, ASA_U_Polar_unit, Fraction_exposed_Native = Native_State(partitionId, partitionSchemes, df, OTnum)
            x = load_atoms_range(partitionId, partitionSchemes, partitionStates, i, df)
            y = calc_ASA_dSconf(x[3], partitionId,partitionStates, partitionSchemes, i, ASA_U_Apolar_unit, ASA_U_Polar_unit, Fraction_exposed_Native, df)
            Fraction_folded = partitionStates[partitionId][i].count(1)/len(partitionStates[partitionId][i])
    

    ASA_N_Apolar, ASA_N_Polar, ASA_N_Apolar_unit, ASA_N_Polar_unit, ASA_U_Apolar_unit, ASA_U_Polar_unit, Fraction_exposed_Native = Native_State(1, partitionSchemes, df, OTnum)
    x = load_atoms_range(1, partitionSchemes, partitionStates, 20, df)
    y = calc_ASA_dSconf(x[3], 1 ,partitionStates, partitionSchemes, 20, ASA_U_Apolar_unit, ASA_U_Polar_unit,ASA_N_Apolar, ASA_N_Polar,Fraction_exposed_Native, df)

    print(y)
    '''

    '''
    ASA_N_Apolar, ASA_N_Polar, ASA_N_Apolar_unit, ASA_N_Polar_unit, ASA_U_Apolar_unit, ASA_U_Polar_unit, Fraction_exposed_Native = Native_State(1, partitionSchemes, df, OTnum)
    x = load_atoms_range(1, partitionSchemes, partitionStates, 20, df)
    y = calc_ASA_dSconf(x[3], 1 ,partitionStates, partitionSchemes, 20, ASA_U_Apolar_unit, ASA_U_Polar_unit,ASA_N_Apolar, ASA_N_Polar,Fraction_exposed_Native, df, stack_from_pdb)

    '''