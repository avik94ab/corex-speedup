# 0: Folded, 1: Unfolded
# num_atoms is the no. of folded atoms
import itertools
import random
import time, sys, math
import pandas as pd
from itertools import product

import numpy as np
from biotite.structure import sasa
import biotite.structure.io as strucio
from multiprocessing import Pool
import matplotlib.pyplot as plt

import tracemalloc

import pdbIO, SFR
from guppy import hpy
'''
THERMODYNAMIC PARAMETERS INITIALIZATION
'''
aCp = 0.44
bCp = -0.26
adeltaH = -8.44
bdeltaH = 31.4
OHCp = 0.17
TsPolar = 335.15
TsApolar = 385.15
dSbb_length_corr = -0.12
exposed_criteria = 0.1
W_Sconf = 1.0  # might be an user input later
Current_Temp = 298.15
ASA_exposed_amide = 7.5
Def_Nprob = 75
RT_Inverse = (1.0 / (1.9872 * Current_Temp))
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

        if (windows[-1][1] - windows[-1][0] + 1 < Minimum_Window_Size):
            windows.remove(windows[-1])
            windows[-1][1] = seq_length - 1

        if (windows[0][1] - windows[0][0] + 1 < Minimum_Window_Size):
            windows.remove(windows[0])
            windows[0][0] = 0

        for w in windows:
            w[0] += 1
            w[1] += 1

        partitionSchemes[partitionId] = windows
        partitionId += 1

    numEnsembles = 0
    for partitionId, partition in partitionSchemes.items():
        print("Partition Scheme No.", partitionId, " contains ", 2 ** len(partition) - 2, " intermediate states")
        unitId = 1
        for segment in partition:
            print("Unit no.", unitId, "Starting residue:", segment[0], " Final residue:", segment[1])
            unitId += 1

        print("-------")
        numEnsembles += 2 ** len(partition) - 2
    print("Total number of micro-states: ", numEnsembles + 2)  # no. of ensemble states +U+F
    print("Total number of partially folded micro-states: ", numEnsembles)
    return partitionSchemes


def state_generator(partitionSchemes):
    '''
    Return dict-> keys: partition scheme no., values: number of folding units
    '''
    partitionStates = {}
    for partitionId, partition in partitionSchemes.items():
        partitionStates[partitionId] = len(partition)
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
            # print(j, end = ":")
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

        # print('------')

    ASA_U_Polar_unit[0] = ASA_U_Polar_unit[0] + 45.0
    j = len(partitionSchemes[partitionId]) - 1
    ASA_U_Apolar_unit[j] = ASA_U_Apolar_unit[j] + 30.0
    ASA_U_Polar_unit[j] = ASA_U_Polar_unit[j] + 30.0 * OTnum

    return ASA_N_Apolar, ASA_N_Polar, ASA_N_Apolar_unit, ASA_N_Polar_unit, ASA_U_Apolar_unit, ASA_U_Polar_unit, Fraction_exposed_Native


def load_atoms_range(partitionId, partitionSchemes, state, df):
    '''
    @param partitionId
    @param partitionSchemes
    @param partitionStates
    @param stateNum
    @param df
    '''
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


def calc_stat_weight(folded_atoms, partitionId, state,  partitionSchemes, ASA_U_Apolar_unit,
                    ASA_U_Polar_unit, ASA_N_Apolar, ASA_N_Polar, Fraction_exposed_Native, df, stack_from_pdb):
    seq_length = int(df['ResNum'].iloc[-1])
    aaConstants = getAAConstants()
    ASA_State_Apolar = 0.0
    ASA_State_Polar = 0.0
    Sum_U_Apolar = 0.0
    Sum_U_Polar = 0.0

    Fraction_exposed = [0.0] * seq_length
    Fraction_amide_exposed = [0.0] * seq_length

    # _, _, pdb_df, _, stack_from_pdb = pdbIO.readPDB('6cne')

    area_folded = pdbIO.asa_calc(stack_from_pdb, folded_atoms)

    for i in folded_atoms:
        element = df.at[i, 'AtomName']
        if element[0] == "C":
            ASA_State_Apolar = ASA_State_Apolar + area_folded[i]
        else:
            ASA_State_Polar = ASA_State_Polar + area_folded[i]

    Sconf = 0.0


    for i in range(len(partitionSchemes[partitionId])):
        if state[i] == 1:
            Sum_U_Apolar = Sum_U_Apolar + ASA_U_Apolar_unit[i]
            Sum_U_Polar = Sum_U_Polar + ASA_U_Polar_unit[i]
            for j in range(partitionSchemes[partitionId][i][0], partitionSchemes[partitionId][i][1] + 1):
                idx = df.index[df['ResNum'] == j].tolist()[0]
                aminoAcid = df.at[idx, 'ResName']
                Sconf = Sconf + aaConstants.at[aminoAcid, 'dSexu'] + aaConstants.at[
                    aminoAcid, 'dSbb'] + dSbb_length_corr + (1.0 - Fraction_exposed_Native[j - 1]) * aaConstants.at[
                            aminoAcid, 'dSbuex']

    for i in range(len(partitionSchemes[partitionId])):
        if state[i] == 0:
            for j in range(partitionSchemes[partitionId][i][0], partitionSchemes[partitionId][i][1] + 1):
                ASA_side_chain = 0.0
                ASA_Amide = 0.0
                k = df.index[df['ResNum'] == j].tolist()
                for atom in k:
                    element = df.at[atom, 'AtomName']
                    if element[0] == "N":  # check with Jamie
                        ASA_Amide = ASA_Amide + area_folded[atom]
                    if element not in ["N", "CA", "C", "O"]:  # check with Jamie
                        ASA_side_chain = ASA_side_chain + area_folded[atom]
                idx = df.index[df['ResNum'] == j].tolist()[0]
                aminoAcid = df.at[idx, 'ResName']
                Fraction_exposed[j - 1] = ASA_side_chain / aaConstants.at[aminoAcid, 'ASAsc']
                Fraction_amide_exposed[j - 1] = ASA_Amide / ASA_exposed_amide
                Sconf = Sconf + (Fraction_exposed[j - 1] - Fraction_exposed_Native[j - 1]) * aaConstants.at[
                    aminoAcid, 'dSbuex']
                # print(j, aminoAcid, ", Sconf = ", Sconf, ", Fraction_exposed[j-1] = ", Fraction_exposed[j-1], ", Fraction_exposed_Native[j-1] = ",Fraction_exposed_Native[j-1],  ", dSbuex = ",aaConstants.at[aminoAcid, 'dSbuex'], "ASA_side_chain = ", ASA_side_chain, "ASA_sc = ", aaConstants.at[aminoAcid, 'ASAsc'])
                # print(j, aminoAcid, ", Sconf = ", Sconf, ", Fraction_exposed[j-1] = ", Fraction_exposed[j-1], ", Fraction_exposed_Native[j-1] = ",Fraction_exposed_Native[j-1],  ", dSbuex = ",aaConstants.at[aminoAcid, 'dSbuex'])

    delASA_ap = ASA_State_Apolar + Sum_U_Apolar - ASA_N_Apolar
    delASA_pol = ASA_State_Polar + Sum_U_Polar - ASA_N_Polar
    dASA_ap = delASA_ap
    dASA_pol = delASA_pol

    SconfN = Sconf * W_Sconf
    dCp = dASA_ap * aCp + dASA_pol * bCp
    dH60 = dASA_ap * adeltaH + dASA_pol * bdeltaH
    dH25 = dH60 + dCp * ((Current_Temp - 273.15) - 60)
    dSsolv_ap25 = dASA_ap * aCp * math.log((Current_Temp / TsApolar))
    dSsolv_pol25 = dASA_pol * bCp * math.log((Current_Temp / TsPolar))
    dG25 = dH25 - Current_Temp * (SconfN + dSsolv_ap25 + dSsolv_pol25)


    return dG25, Sconf, delASA_ap, delASA_pol


def sampleMicrostate(args):

    partitionSchemes, partitionStates, df, stack_from_pdb, OTnum, factor, Native_State_dct = args

    selection = -99999999
    selRand = 99999999

    count = 0

    start_time = time.time()

    while(selRand>selection):

        partitionId = np.random.randint(1, len(partitionSchemes) + 1)
        state = np.random.randint(2, size = partitionStates[partitionId])


        ASA_N_Apolar, ASA_N_Polar, ASA_N_Apolar_unit, ASA_N_Polar_unit, ASA_U_Apolar_unit, ASA_U_Polar_unit, Fraction_exposed_Native = Native_State_dct[partitionId]
        state, num_atoms, num_residues, folded_atoms = load_atoms_range(partitionId, partitionSchemes, state, df)
        dG25, Sconf, delASA_ap, delASA_pol = calc_stat_weight(folded_atoms, partitionId, state, partitionSchemes, ASA_U_Apolar_unit,
                            ASA_U_Polar_unit, ASA_N_Apolar, ASA_N_Polar, Fraction_exposed_Native, df, stack_from_pdb)
        dG = dG25
        if (dG < 0.0):
            dG=0.0
        dG = dG * factor
        selection = math.exp(-dG * RT_Inverse)
        selRand = random.random()

        count += 1




    Fraction_folded = list(state).count(0) / len(state)

    stateFlag = ""
    for f in state:
        stateFlag += str(f)

    end_time = time.time()

    return partitionId, Fraction_folded, Sconf, delASA_ap, delASA_pol, stateFlag, count, end_time - start_time



if __name__ == '__main__':

    # fileSize, seq_length, df = readPDBinfo("1ediA.pdb.info")

    # seq_length, OTnum, df = infoStable.trial()

    # seq_length, OTnum, pdb_lst, df, stack_from_pdb = pdbIO.readPDB('6cne')

    # print(type(df.at[1, 'ResNum']))

    pdbID = "6cne"
    window_size = 5
    Minimum_Window_Size = 4
    sampleSize = 18300 * 3

    # pdbID = str(sys.argv[1])
    # window_size = int(sys.argv[2])
    # Minimum_Window_Size = int(sys.argv[3])
    # sampleSize = int(sys.argv[4])

    seq_length, OTnum, pdb_lst, df, stack_from_pdb = pdbIO.readPDB(pdbID)

    # print(df.index[df['ResNum'] == 56].tolist()[0])

    # print(type(df.at[1, 'ResNum']))

    partitionSchemes = partition_generator(seq_length, window_size, Minimum_Window_Size)

    partitionStates = state_generator(partitionSchemes)

    # TODO: Insert OTnum from readPDBfile

    output = []

    args = []  # list of all possible arguement combination
    '''
    for partitionId, partition in partitionStates.items():
        for i in range(len(partition)):
            args.append((partitionSchemes, partitionStates, df, stack_from_pdb, OTnum))
            
    
    '''

    Native_State_dct = {}

    for i in range(len(partitionSchemes)):
        Native_State_dct[i+1] = Native_State(i+1, partitionSchemes, df, OTnum)


    partitionId = len(partitionSchemes)
    state = [1] * partitionStates[partitionId]



    ASA_N_Apolar, ASA_N_Polar, ASA_N_Apolar_unit, ASA_N_Polar_unit, ASA_U_Apolar_unit, ASA_U_Polar_unit, Fraction_exposed_Native = Native_State_dct[partitionId]
    state, num_atoms, num_residues, folded_atoms = load_atoms_range(partitionId, partitionSchemes, state, df)
    dG25, Sconf, delASA_ap, delASA_pol = calc_stat_weight(folded_atoms, partitionId, state, partitionSchemes,
                                                          ASA_U_Apolar_unit, ASA_U_Polar_unit,
                                                          ASA_N_Apolar, ASA_N_Polar, Fraction_exposed_Native, df,
                                                          stack_from_pdb)


    print("dG25 for fully unfolded state = ", dG25)
    Nprob = Def_Nprob
    prob = 0.01 * Nprob
    print("prob value = ", prob)
    factor = -math.log(prob) / (dG25 * RT_Inverse)
    if (dG25 < 5000.0):
        factor = -math.log(prob) / (5000.0 * RT_Inverse)
    print("factor = ", factor)

    for i in range(sampleSize):
        args.append((partitionSchemes, partitionStates, df, stack_from_pdb, OTnum, factor, Native_State_dct))


    start_time = time.time()

    tracemalloc.start()
    with Pool(2) as pool:
        result = pool.map(sampleMicrostate, args)

    # displaying the memory
    print(tracemalloc.get_traced_memory())

    # stopping the library
    tracemalloc.stop()


    print("\n--- %s seconds---" % (time.time() - start_time))


    num_tries = []
    timeTaken = []
    for r in result:
        num_tries.append(r[-2])
        timeTaken.append(r[-1])


    plt.scatter(num_tries,timeTaken)
    plt.xlabel("No. of tries")
    plt.ylabel("Time taken (seconds)")
    plt.xticks(np.arange(1, max(num_tries) + 1, 1.0))

    plt.plot()
    plt.savefig("Profiling.pdf")

    plt.hist(num_tries, np.arange(1, max(num_tries) + 1, 1.0))
    plt.xlabel("No. of tries")
    plt.ylabel("Number of samples")
    plt.plot()
    plt.savefig("HistogramSamples.pdf")



    print("Total GOOD states sampled (user-defined): ", len(result))
    print("Total states sampled: ", sum(num_tries))
    print("Total unique states sampled: ", len(set(result)))

    tmpStr = []
    for r in result:
        if r != None:
            tmpStr.append(
                str(r[0]) + ' ' + str(r[1]) + ' ' + str(r[2]) + ' ' + str(
                    r[3]) + ' ' + str(r[4]) + ' ' + str(r[5]) + '\n')
    output = "".join(tmpStr)

    text_file = open(pdbID + ".pdb." + str(window_size) + "." + str(Minimum_Window_Size), "wt")
    n = text_file.write(output)
    text_file.close()



    Partition_Function25 = 1.0

    text_file = open(pdbID + ".pdb." + str(window_size) + "." + str(Minimum_Window_Size), 'r')
    ensemble = text_file.readlines()
    text_file.close()

    tmp = []

    for e in result:
        e = list(e)
        e = e[0:6]
        del e[1]
        tmp.append(e)

    ensemble = tmp

    ln_kf = SFR.Residue_Probabilities(ensemble, seq_length, partitionSchemes, Partition_Function25)

    corexOut = []
    for i in range(len(ln_kf)):
        corexOut.append(str(i + 1) + " " + str(ln_kf[i]) + "\n")
    output = "".join(corexOut)

    file = open(pdbID + ".pdb." + str(window_size) + "." + str(Minimum_Window_Size) + ".ThermoDescriptMC", 'wt')
    n = file.write(output)
    file.close()

