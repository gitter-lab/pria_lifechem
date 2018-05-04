import numpy as np
import os
import pandas as pd

in_dir = './job_results_pred/'
out_dir = './top_250/'

classes = os.listdir(in_dir)
stage = 'stage_2'
keck_file = './datasets/keck_lc4_Nov_6_2017.csv'
df = pd.read_csv(keck_file)
actives_df = df[df.Keck_Pria_AS_Retest==1]
n_tests = 250

req_columns = ['Molecule', 'SMILES', 'Fingerprints', 'Keck_Pria_AS_Retest',
               'Keck_Pria_Continuous']
rename_columns = ['Molecule', 'SMILES', 'Fingerprints', 
           'PriA-SSB AS',
           'PriA-SSB Continuous TRUE']
                            
for c in classes:
    model_dirs = os.listdir(in_dir+'/'+c+'/'+stage+'/')
    for m in model_dirs:
        file_name = in_dir+'/'+c+'/'+stage+'/'+m+'/fold_0.npz'
        fold_np = np.load(file_name)
        labels, y_tr, y_v, y_test, y_pred_on_train, y_pred_on_val, y_pred_on_test = fold_np['labels'], fold_np['y_train'], fold_np['y_val'], fold_np['y_test'], fold_np['y_pred_on_train'], fold_np['y_pred_on_val'], fold_np['y_pred_on_test']
        labels = [e.decode('UTF-8') for e in labels]
        
        y_test = y_test[:,0]
        y_pred_on_test = y_pred_on_test[:,0]
        
        sorted_indices = np.argsort(y_pred_on_test, axis=0)[::-1][:n_tests]
                
        y_test = pd.DataFrame(data=y_test[sorted_indices], index=list(sorted_indices))
        y_pred_on_test = pd.DataFrame(data=y_pred_on_test[sorted_indices], index=list(sorted_indices))
        molecule_ids = df['Molecule'][sorted_indices]
        molecule_ids = molecule_ids[(y_test == 1).index[np.where(y_test == 1)[0]]]

        common  = pd.merge(actives_df, molecule_ids.to_frame(), on='Molecule')       
        actives_df = actives_df[(~actives_df.Molecule.isin(common.Molecule))]
        
actives_df = actives_df[req_columns]
actives_df.columns = rename_columns

save_file = out_dir+'actives_not_covered.csv'
actives_df.to_csv(save_file, index=False)