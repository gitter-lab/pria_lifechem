import numpy as np
import os
import pandas as pd

in_dir = './job_results_pred/'
out_dir = './top_250/'

classes = os.listdir(in_dir)
stage = 'stage_2'
keck_file = './datasets/keck_lc4_Nov_6_2017.csv'
df = pd.read_csv(keck_file)
n_tests = 250

for c in classes:
    model_dirs = os.listdir(in_dir+'/'+c+'/'+stage+'/')
    if not os.path.exists(out_dir+'/'+c+'/'):
        os.makedirs(out_dir+'/'+c+'/')
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
        smiles = df['SMILES'][sorted_indices]
        continuous = df['Keck_Pria_Continuous'][sorted_indices]
        fps = df['Fingerprints'][sorted_indices]
        
        model_df = pd.concat([molecule_ids, smiles, fps, y_test, y_pred_on_test, continuous],
                             axis=1, ignore_index=True)
        model_df.columns = ['Molecule', 'SMILES', 'Fingerprints', 
                            'PriA-SSB AS TRUE', 'PriA-SSB AS PREDICTED',
                            'PriA-SSB Continuous TRUE']
                     
        save_dir = out_dir+'/'+c+'/'+m+'/'
        save_file = save_dir+'/top_250.csv'
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        model_df.to_csv(save_file, index=False)