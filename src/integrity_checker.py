# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 07:41:32 2017

@author: Moeman
"""


"""
 Script for checking md5 hash of all files. Adapted from this post:
 www.quora.com/How-do-you-create-a-checksum-of-a-file-in-Python
 
 Usage: Pass this function the directory containing the fold files. 
        Currently only supports 3, 4, and 5 folds via hard-coded hash keys.
"""
import hashlib
import os

def check_keck_fold_integrity(fold_directory):
    #Add to this dictionary to add more fold options    
    switch = {3 : fold_3_keys,
              4 : fold_4_keys,
              5 : fold_5_keys
              } 
              
    file_list = os.listdir(fold_directory)
    file_list = sorted(file_list)
    k = len(file_list)
    key_hash_list = None
    
    try:
        key_hash_list = switch[k]()
    except KeyError:
        print("Number of folds is not supported. Check for repo updates.")
        return
    
    owd = os.getcwd()
    os.chdir(fold_directory)
    md5_hash_list = [(fname, hashlib.md5(open(fname, 'rb').read()).digest()) for fname in file_list]
    os.chdir(owd)
    
    no_errors = True
    for i in range(k):
        if key_hash_list[i] != md5_hash_list[i]:
            no_errors = False
          
    if no_errors:
        print("All files correct with no errors")
    else:
        print("There are errors in the files")
        

def fold_3_keys():
    return [('file_0.csv', b'\xc2)4\x85\x8f\xf5\x17Wm\xcf\xbbG\x9a\x81\x03\xa6'),
     ('file_1.csv', b'\xdd<M\xf1\xf43+\xe5\xf7\xa6\xc3:\xf2X5\xca'),
     ('file_2.csv', b'\x93r\xa8\xd0\x8d\xecS\x058j\xc0\x05\x04<\xae\xba')]
    
def fold_4_keys():
    return [('file_0.csv', b'\xd5\x01\x9d\x9eC9\xba\xc1\x08\xb9\x7fjp\x02; '),
     ('file_1.csv', b'\x9e\x04\xce\xca\x92\xc8\xc5J\x81\xb2m\xc6)\x16?\x9a'),
     ('file_2.csv', b'W\xd9\x1c\x16R\x06\xe8\xba\xe9$\xdew\xe3\x01\xe8E'),
     ('file_3.csv', b'q\x02\xee\xc7x,\xb4|"L\x10\x98EI\xd1\xfe')]

def fold_5_keys():
    return [('file_0.csv', b'\xfb\x97\xd0\xdcs\xd5\x12^\xfc\x96\xef1\\\x03\x0e\n'),
     ('file_1.csv', b'\x11F\xf9\xa3v2\xcaW\xf0\xb5\xc8\x1dZ\xae\x0c\xb6'),
     ('file_2.csv', b'\x05\xfd\t^\x18_M\xcaJ\xc6\x81\xc8PE\xed\x06'),
     ('file_3.csv', b'\xaa0LG\xc3\xe0\xe1y\xf0\xe1G\x84\xc9uz\xed'),
     ('file_4.csv', b'\x94\xf1\xdfw\x81\x89/,\x94\x12\x9f\x9c\xc1\xc2.\xe8')]