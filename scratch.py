import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from biotite.structure import sasa
import biotite.structure.io as strucio
from main import readPDBinfo
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np


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

stack_from_pdb = strucio.load_structure("1ediA.pdb")


vdw_radii = []
for idx in range(len(stack_from_pdb)):
    vdw_radii.append(radius_table[stack_from_pdb.get_atom(idx).atom_name])
vdw_radii = np.array(vdw_radii)

atom_sasa_exp = sasa(stack_from_pdb, point_number = 1000, vdw_radii= vdw_radii)
print("Done")

'''
fileSize, seq_length, df = readPDBinfo("1ediA.pdb.info")
atom_sasa_hilser = pd.to_numeric(df["Nat.Area"]).tolist()

corr, _ = pearsonr(atom_sasa_hilser, atom_sasa_exp)

plt.scatter(atom_sasa_hilser, atom_sasa_exp)
plt.xlabel("Nat.Area(Hilser)")
plt.ylabel("Nat.Area(Experimental)")
plt.legend(title = "Pearson correlation = " + str(round(corr,4)))
plt.savefig("sample.pdf")

'''









