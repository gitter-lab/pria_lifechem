import pandas as pd
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem


def transform_data(input_file_name, output_file_name, updated_binary_label):
    suppl = Chem.SDMolSupplier(input_file_name)
    output_file = open(output_file_name, 'w')
    good = 0
    bad = 0
    i = 0
    unique = set()
    print >> output_file, "molecule ID(RegID),library,existing SMILES,generated SMILES,1024_fingerprint,Pria_SSB_%INH,true_label"
    for mol in suppl:
        i += 1
        if mol is None:
            bad += 1
            continue
        name = mol.GetProp("RegID")
        if name in unique:
            continue
        unique.add(name)
        good += 1
        library = mol.GetProp("Library")
        existing_smiles = mol.GetProp("SMILES")
        generated_smiles = Chem.CanonSmiles(Chem.MolToSmiles(mol))
        generated_mol = Chem.MolFromSmiles(existing_smiles)
        fingerprints = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)

        pria = mol.GetProp("Pria_SSB_%INH")
        pria = float(pria)

        true_label = updated_binary_label[updated_binary_label['Molecule'] == name]['Keck_Pria_AS_Retest']
        true_label = true_label.tolist()[0]
        output_line = "{},{},{},{},{},{},{}".format(name,
                                                    library,
                                                    existing_smiles,
                                                    generated_smiles,
                                                    fingerprints.ToBitString(),
                                                    pria,
                                                    true_label)
        print >> output_file, output_line
    print("{} items in all, {} are good, {} are bad".format(i, good, bad))
    return


if __name__ == '__main__':
    discrete_file = pd.ExcelFile('../dataset/screening_smsf_actives.xlsx')
    updated_binary_label = discrete_file.parse('Keck_Pria_Retest')
    transform_data('../dataset/lc123_keckdata.sdf', '../dataset/keck_complete.csv', updated_binary_label)