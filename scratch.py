from Bio.PDB import PDBIO, PDBParser

from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue


# to work with some non orthodox pdbs
import warnings
warnings.filterwarnings('ignore')


io = PDBIO()
parser = PDBParser()


# my_pdb_structure = parser.get_structure('test', 'test.pdb')
my_pdb_structure = parser.get_structure('test', '2qle_stripped.pdb')

print(my_pdb_structure)


# renumber residue in my_pdb_structure
residue_N = 1
for model in my_pdb_structure:
    for chain in model:
            for residue in chain:
                print(residue.id)
                if 'A' in residue.id[2]:
                    residue.id = (residue.id[0], residue_N-1, residue.id[2])
                    print('----',residue.id)

                else:
                    residue.id = (residue.id[0], residue_N, residue.id[2])
                    print('----',residue.id)
                    residue_N += 1


# this bit just print the renumbered my_pdb_structure
print('\n stucture with renumbered atoms : \n___________________________________')
for model in my_pdb_structure:
    for chain in model:
            for residue in chain:
                print(residue, residue.id)


io.set_structure(my_pdb_structure)
# io.save('renumbered.pdb')
io.save('2qle.pdb',  preserve_atom_numbering=True)