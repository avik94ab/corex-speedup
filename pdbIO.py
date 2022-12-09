from biotite.structure import sasa
from biotite.structure.io.pdb import PDBFile
import numpy as np
from biopandas.pdb import PandasPdb
import pandas as pd

#dictionary of VDW radii
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

def stripPDB(fileName):

    ppdb = PandasPdb().read_pdb(fileName+'.pdb')

    # Remove all HETATM entries without MSE
    ppdb.df['HETATM'] = ppdb.df['HETATM'][ppdb.df['HETATM']['residue_name'] == 'MSE']

    # Replace SE of MSE by SD
    ppdb.df['HETATM'].loc[ppdb.df['HETATM']['atom_name'] == 'SE', 'atom_name'] = 'SD'

    # Change all MSE to MET
    ppdb.df['HETATM'].loc[ppdb.df['HETATM']['residue_name'] == 'MSE', 'residue_name'] = 'MET'

    # Remove all Hydrogen entries
    ppdb.df['ATOM'] = ppdb.df['ATOM'][ppdb.df['ATOM']['element_symbol'] != 'H']
    ppdb.df['HETATM'] = ppdb.df['HETATM'][ppdb.df['HETATM']['element_symbol'] != 'H']

    # Only keep chain A entries
    ppdb.df['ATOM'] = ppdb.df['ATOM'][ppdb.df['ATOM']['chain_id'] == 'A']
    ppdb.df['HETATM'] = ppdb.df['HETATM'][ppdb.df['HETATM']['chain_id'] == 'A']

    # Change remanining HETATM to ATOM
    ppdb.df['HETATM'].loc[ppdb.df['HETATM']['record_name'] == 'HETATM', 'record_name'] = 'ATOM'

    # for atoms with multiple occupancy, only consider the first density
    x = ppdb.df['ATOM']['occupancy'].tolist()
    y = ppdb.df['ATOM']['element_symbol'].tolist()
    z = ppdb.df['ATOM']['atom_name'].tolist()

    # Remove all ANISOU entries and save stripped down file
    ppdb.to_pdb(path=fileName + "_stripped.pdb",
                records=['ATOM', 'HETATM', 'OTHERS'],
                gz=False,
                append_newline=False)

def readPDB(fileName):
    #get stripped version of .pdb file
    stripPDB(fileName)
    file = PDBFile.read(fileName + "_stripped.pdb")
    stack_from_pdb = file.get_structure(altloc='first').get_array(0)
    vdw_radii = []
    for idx in range(len(stack_from_pdb)):
        vdw_radii.append(radius_table[stack_from_pdb.get_atom(idx).atom_name])
    vdw_radii = np.array(vdw_radii)

    #calculate ASA using Shrake-Rupley algorithm
    atom_sasa_exp = sasa(stack_from_pdb, point_number=1000, vdw_radii=vdw_radii)

    atom_lst = []
    OTnum = 0

    for idx in range(len(stack_from_pdb)):
        atom_lst.append((idx, stack_from_pdb.get_atom(idx).res_id, stack_from_pdb.get_atom(idx).res_name, stack_from_pdb.get_atom(idx).atom_name,
                     stack_from_pdb.get_atom(idx).coord, vdw_radii[idx], atom_sasa_exp[idx]))
        if stack_from_pdb.get_atom(idx).atom_name == "OXT" or stack_from_pdb.get_atom(idx).atom_name == "OT":
            OTnum += 1

    #atom_lst = pd.DataFrame.from_records(atom_lst,columns=['AtomNum', 'ResNum', 'ResName', 'AtomName', 'xyz', 'Radius', 'Nat.Area'])

    return OTnum, atom_lst


OTnum, atom_lst = readPDB('1ediA')

print(atom_lst)





