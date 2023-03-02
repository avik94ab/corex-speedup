from numba import njit
import math, time, pdbIO
from numba import jit

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
ASA_exposed_amide = 7.5


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
    k_f = [0.0] * seq_length
    lnk_f = [0.0] * seq_length


    for e in ensemble:
        print(e.split())
        partitionId = int(e.split()[0])
        Sconf = float(e.split()[2])
        delASA_ap = float(e.split()[3])
        delASA_pol = float(e.split()[4])
        state = [int(i) for i in e.split()[5]]




        dG25, stat_weight, dH_ap, dH_pol, dSconf, dG_solv, Partition_Function25 = calc_component_energies(Sconf, delASA_ap, delASA_pol, Partition_Function25)

        for i in range(len(partitionSchemes[partitionId])):
            if state[i] == 1:
                for j in range(partitionSchemes[partitionId][i][0], partitionSchemes[partitionId][i][1] + 1):
                    Prob_unfolded25[j-1] = Prob_unfolded25[j-1] + stat_weight



    print("Partition Function 25 = ", Partition_Function25)

    for i in range(seq_length):
        Prob_unfolded25[i-1] = Prob_unfolded25[i-1] / Partition_Function25

    for i in range(seq_length):

        k_f[i-1] = (1.0 - Prob_unfolded25[i-1]) / Prob_unfolded25[i-1]
        lnk_f[i-1] = math.log(k_f[i-1])


    return lnk_f









