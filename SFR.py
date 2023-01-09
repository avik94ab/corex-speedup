from numba import njit
import math, time, pdbIO
from numba import jit
from EnsembleGenerator import partition_generator, state_generator
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
#TODO:
# Ask jamie about the value allocation
W_Sconf=0.5 #can be made user-defined
Current_Temp = 298.15
ASA_exposed_amide=7.5

@jit(nopython=True)
def calc_component_energies(Sconf, delASA_ap, delASA_pol, Partition_Function25):

    SconfN = Sconf * W_Sconf
    dH_ap = delASA_ap * (adeltaH + aCp * ((Current_Temp - 273.15) - 60))
    dH_pol = delASA_pol * (bdeltaH + bCp * ((Current_Temp - 273.15) - 60))
    dSconf = SconfN
    dSsolv_ap = delASA_ap * aCp * math.log((Current_Temp / TsApolar))
    dSsolv_pol = delASA_pol * bCp * math.log((Current_Temp / TsPolar))
    dG_solv = dH_ap + dH_pol - Current_Temp * (dSsolv_ap + dSsolv_pol)
    dG25 = dG_solv - Current_Temp * SconfN
    stat_weight = math.exp(-dG25 / (1.9872 * Current_Temp))
    Partition_Function25 = Partition_Function25 + stat_weight #ask Jamie, does its value change over time?
    return dG25, stat_weight, dH_ap, dH_pol, dSconf, dG_solv, Partition_Function25

def Residue_Probabilities(ensemble, seq_length, partitionSchemes, partitionStates, Partition_Function25):
    Prob_unfolded25 = [0.0] * seq_length
    #dH_pol_nf = [0.0]  * seq_length
    #dH_ap_nf = [0.0] * seq_length
    #dSconf_nf = [0.0] * seq_length
    #dG_solv_nf = [0.0] * seq_length
    #dH_pol_f = [0.0] * seq_length
    #dH_ap_f = [0.0] * seq_length
    #dSconf_f = [0.0] * seq_length
    #dG_solv_f = [0.0] * seq_length
    #deltaH_pol = [0.0] * seq_length
    #deltaH_ap = [0.0] * seq_length
    #deltaSconf = [0.0] * seq_length
    #deltaG_solv = [0.0] * seq_length
    #H_ratio = [0.0] * seq_length
    #S_ratio = [0.0] * seq_length
    k_f = [0.0] * seq_length
    lnk_f = [0.0] * seq_length
    #Part_Funct_nf = [0.0] * seq_length
    #Part_Funct_f = [0.0] * seq_length

    for e in ensemble:
        partitionId = int(e.split()[0])


        Sconf = float(e.split()[4])
        delASA_ap = float(e.split()[5])
        delASA_pol = float(e.split()[6])
        state = [int(i) for i in e.split()[7]]




        dG25, stat_weight, dH_ap, dH_pol, dSconf, dG_solv, Partition_Function25 = calc_component_energies(Sconf, delASA_ap, delASA_pol, Partition_Function25)

        for i in range(len(partitionSchemes[partitionId])):
            if state[i] == 1:
                for j in range(partitionSchemes[partitionId][i][0], partitionSchemes[partitionId][i][1] + 1):
                    Prob_unfolded25[j-1] = Prob_unfolded25[j-1] + stat_weight
                    #dH_pol_nf[j-1] = dH_pol_nf[j-1] + dH_pol * stat_weight
                    #dH_ap_nf[j-1] = dH_ap_nf[j-1] + dH_ap * stat_weight
                    #dSconf_nf[j-1] = dSconf_nf[j-1] + dSconf * stat_weight
                    #dG_solv_nf[j-1] = dG_solv_nf[j-1] + dG_solv * stat_weight
            if state[i] == 0:
                for j in range(partitionSchemes[partitionId][i][0], partitionSchemes[partitionId][i][1] + 1):
                    pass
                    #dH_pol_f[j-1] = dH_pol_f[j-1] + dH_pol * stat_weight
                    #dH_ap_f[j-1] = dH_ap_f[j-1] + dH_ap * stat_weight
                    #dSconf_f[j-1] = dSconf_f[j-1] + dSconf * stat_weight
                    #dG_solv_f[j-1] = dG_solv_f[j-1] + dG_solv * stat_weight

    print("Partition Function 25 = ", Partition_Function25)

    for i in range(seq_length):
        #Part_Funct_nf[i-1] = Prob_unfolded25[i-1]
        #Part_Funct_f[i-1] = Partition_Function25 - Part_Funct_nf[i-1]
        #deltaH_pol[i-1] = dH_pol_f[i-1] / Part_Funct_f[i-1] - dH_pol_nf[i-1] / Part_Funct_nf[i-1]
        #deltaH_ap[i-1] = dH_ap_f[i-1] / Part_Funct_f[i-1] - dH_ap_nf[i-1] / Part_Funct_nf[i-1]
        #deltaSconf[i-1] = dSconf_f[i-1] / Part_Funct_f[i-1] - dSconf_nf[i-1] / Part_Funct_nf[i-1]
        #deltaG_solv[i-1] = dG_solv_f[i-1] / Part_Funct_f[i-1] - dG_solv_nf[i-1] / Part_Funct_nf[i-1]
        Prob_unfolded25[i-1] = Prob_unfolded25[i-1] / Partition_Function25

    for i in range(seq_length):
        #H_ratio[i-1] = deltaH_pol[i-1] / deltaH_ap[i-1]
        #S_ratio[i-1] = deltaSconf[i-1] / deltaG_solv[i-1]
        k_f[i-1] = (1.0 - Prob_unfolded25[i-1]) / Prob_unfolded25[i-1]
        lnk_f[i-1] = math.log(k_f[i-1])


    return lnk_f


if __name__== '__main__':




    states = []
    file = open('tmp.txt', 'r')
    ensemble = file.readlines()
    for e in ensemble:
        units = []
        microstate = e.split()
        for m in microstate[4]:
            units.append(int(m))
        states.append(units)
    #print(states)

    seq_length, OTnum, pdb_lst, df, stack_from_pdb = pdbIO.readPDB('6cne')

    partitionSchemes = partition_generator(seq_length, 5, 4)
    partitionStates = state_generator(partitionSchemes)

    Partition_Function25 = 1.0

    file = open('ensemble_output.txt', 'r')
    ensemble = file.readlines()

    Residue_Probabilities(ensemble, seq_length, partitionSchemes, partitionStates, Partition_Function25)

    Prob_unfolded25 = [0.0] * seq_length
    #dH_pol_nf = [0.0]  * seq_length
    #dH_ap_nf = [0.0] * seq_length
    #dSconf_nf = [0.0] * seq_length
    #dG_solv_nf = [0.0] * seq_length
    #dH_pol_f = [0.0] * seq_length
    #dH_ap_f = [0.0] * seq_length
    #dSconf_f = [0.0] * seq_length
    #dG_solv_f = [0.0] * seq_length
    #deltaH_pol = [0.0] * seq_length
    #deltaH_ap = [0.0] * seq_length
    #deltaSconf = [0.0] * seq_length
    #deltaG_solv = [0.0] * seq_length
    #H_ratio = [0.0] * seq_length
    #S_ratio = [0.0] * seq_length
    k_f = [0.0] * seq_length
    lnk_f = [0.0] * seq_length
    #Part_Funct_nf = [0.0] * seq_length
    #Part_Funct_f = [0.0] * seq_length

    '''

    states = []
    for partitionId, partitions in partitionStates.items():
        for i in range(len(partitions)):
            state = partitionStates[partitionId][i]
            states.append(state)

    '''

    Partition_Function25 = 1.0
    file = open('6cne.pdb.5.4.sfe', 'r')
    ensemble = file.readlines()
    i = 0
    for i in range(len(ensemble)):
        microstate = ensemble[i].split()[0:7]
        microstate.append(states[i])

        Sconf = float(microstate[4])
        delASA_ap = float(microstate[5])
        delASA_pol = float(microstate[6])
        state = microstate[7]

        dG25, stat_weight, dH_ap, dH_pol, dSconf, dG_solv, Partition_Function25 = calc_component_energies(Sconf, delASA_ap, delASA_pol, Partition_Function25)

        partitionId = int(microstate[0])



        for i in range(len(partitionSchemes[partitionId])):
            if state[i] == 1:
                for j in range(partitionSchemes[partitionId][i][0], partitionSchemes[partitionId][i][1] + 1):
                    Prob_unfolded25[j-1] = Prob_unfolded25[j-1] + stat_weight
                    #dH_pol_nf[j-1] = dH_pol_nf[j-1] + dH_pol * stat_weight
                    #dH_ap_nf[j-1] = dH_ap_nf[j-1] + dH_ap * stat_weight
                    #dSconf_nf[j-1] = dSconf_nf[j-1] + dSconf * stat_weight
                    #dG_solv_nf[j-1] = dG_solv_nf[j-1] + dG_solv * stat_weight
            if state[i] == 0:
                for j in range(partitionSchemes[partitionId][i][0], partitionSchemes[partitionId][i][1] + 1):
                    pass
                    #dH_pol_f[j-1] = dH_pol_f[j-1] + dH_pol * stat_weight
                    #dH_ap_f[j-1] = dH_ap_f[j-1] + dH_ap * stat_weight
                    #dSconf_f[j-1] = dSconf_f[j-1] + dSconf * stat_weight
                    #dG_solv_f[j-1] = dG_solv_f[j-1] + dG_solv * stat_weight

    print("Partition Function 25 = ", Partition_Function25)

    for i in range(seq_length):
        #Part_Funct_nf[i-1] = Prob_unfolded25[i-1]
        #Part_Funct_f[i-1] = Partition_Function25 - Part_Funct_nf[i-1]
        #deltaH_pol[i-1] = dH_pol_f[i-1] / Part_Funct_f[i-1] - dH_pol_nf[i-1] / Part_Funct_nf[i-1]
        #deltaH_ap[i-1] = dH_ap_f[i-1] / Part_Funct_f[i-1] - dH_ap_nf[i-1] / Part_Funct_nf[i-1]
        #deltaSconf[i-1] = dSconf_f[i-1] / Part_Funct_f[i-1] - dSconf_nf[i-1] / Part_Funct_nf[i-1]
        #deltaG_solv[i-1] = dG_solv_f[i-1] / Part_Funct_f[i-1] - dG_solv_nf[i-1] / Part_Funct_nf[i-1]
        Prob_unfolded25[i-1] = Prob_unfolded25[i-1] / Partition_Function25

    for i in range(seq_length):
        #H_ratio[i-1] = deltaH_pol[i-1] / deltaH_ap[i-1]
        #S_ratio[i-1] = deltaSconf[i-1] / deltaG_solv[i-1]
        k_f[i-1] = (1.0 - Prob_unfolded25[i-1]) / Prob_unfolded25[i-1]
        lnk_f[i-1] = math.log(k_f[i-1])




        #dG25, stat_weight, dH_ap, dH_pol, dSconf, dG_solv, Partition_Function25 = calc_component_energies(microstate, Partition_Function25)
    #print(Partition_Function25)










