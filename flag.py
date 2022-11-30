from biopandas.pdb import PandasPdb
ppdb = PandasPdb().read_pdb('./1ediA.pdb')



#Remove all HETATM entries without MSE
ppdb.df['HETATM'] = ppdb.df['HETATM'][ppdb.df['HETATM']['residue_name']=='MSE']

#Replace SE of MSE by SD
ppdb.df['HETATM'].loc[ppdb.df['HETATM']['atom_name'] == 'SE', 'atom_name'] = 'SD'

#Change all MSE to MET
ppdb.df['HETATM'].loc[ppdb.df['HETATM']['residue_name'] == 'MSE', 'residue_name'] = 'MET'

#Remove all Hydrogen entries
ppdb.df['ATOM'] = ppdb.df['ATOM'][ppdb.df['ATOM']['element_symbol']!='H']
ppdb.df['HETATM'] = ppdb.df['HETATM'][ppdb.df['HETATM']['element_symbol']!='H']

#Only keep chain A entries
ppdb.df['ATOM'] = ppdb.df['ATOM'][ppdb.df['ATOM']['chain_id']=='A']
ppdb.df['HETATM'] = ppdb.df['HETATM'][ppdb.df['HETATM']['chain_id']=='A']

#Change remanining HETATM to ATOM
ppdb.df['HETATM'].loc[ppdb.df['HETATM']['record_name'] == 'HETATM', 'record_name'] = 'ATOM'


#for atoms with multiple occupancy, only consider the first density
x = ppdb.df['ATOM']['occupancy'].tolist()
y = ppdb.df['ATOM']['element_symbol'].tolist()
z = ppdb.df['ATOM']['atom_name'].tolist()


for i in range(len(x)):
    print(z[i])


#Remove all ANISOU entries
ppdb.to_pdb(path='./6cne_stripped.pdb',
            records=['ATOM','HETATM' ,'OTHERS'],
            gz=False,
            append_newline=False)