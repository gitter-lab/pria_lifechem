import argparse
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '..')  # Add path from parent folder
sys.path.insert(0, '.')  # Add path from current folder
import os
from evaluation import *

"""
    Runs evaluation metrics on the 100 RF_h Y-Scrambled prediction results.
    
    Note: uses conda_env.yml python2 environment to generate metrics results.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--pred_dir', action="store", dest="pred_dir", required=True)
parser.add_argument('--output_dir', action="store", dest="output_dir", required=True)
parser.add_argument('--process_num', action="store", dest="process_num", required=True)
#####
given_args = parser.parse_args()
pred_dir = given_args.pred_dir
output_dir = given_args.output_dir
process_num = int(given_args.process_num)
#####
# load preds for the process_num'th Y-Scrambled model
npz_file_name = pred_dir+'process_{}.npz'.format(process_num)
preds_npz = np.load(npz_file_name)
labels, y_train, y_test, y_pred_on_train, y_pred_on_test = preds_npz['labels'], preds_npz['y_train'], preds_npz['y_test'], preds_npz['y_pred_on_train'], preds_npz['y_pred_on_test']
labels = [e.decode('UTF-8') for e in labels]

output_dir = output_dir+'/process_{}/fold_{}/'.format(process_num, 0)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(output_dir+'/train_metrics/'):
    os.makedirs(output_dir+'/train_metrics/')
if not os.path.exists(output_dir+'/val_metrics/'):
    os.makedirs(output_dir+'/val_metrics/')
if not os.path.exists(output_dir+'/test_metrics/'):
    os.makedirs(output_dir+'/test_metrics/')

# evaluate on the training set
evaluate_model(y_train, y_pred_on_train, output_dir+'/train_metrics/', labels)
# evaluate on the prospective set
evaluate_model(y_test, y_pred_on_test, output_dir+'/test_metrics/', labels)