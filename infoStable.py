import pandas as pd
import pdbIO

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



def trial():
    fileSize, seq_length, df = readPDBinfo('6cne.pdb.info')

    OTnum, atom_lst = pdbIO.readPDB('6cne')

    output = []
    idx = 0


    for idx in range(fileSize):
        output.append((idx, df.iloc[idx]['ResNum'], df.iloc[idx]['ResName'], df.iloc[idx]['AtomName'], atom_lst[idx][4], radius_table[df.iloc[idx]['AtomName']], df.iloc[idx]['Nat.Area']))

    for out in output:
        print(out)

    df = pd.DataFrame.from_records(output, columns=['AtomNum', 'ResNum', 'ResName', 'AtomName', 'xyz', 'Radius', 'Nat.Area'])

    return OTnum, df


