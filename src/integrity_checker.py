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
        
     dataset_name has three options 'keck', 'pcba', and 'keck_pcba'
"""
import hashlib
import re
import os

def check_keck_fold_integrity(fold_directory, dataset_name='keck'):
    #Add to this dictionary to add more fold options    
    switch = {3 : fold_3_keys,
              4 : fold_4_keys,
              5 : fold_5_keys
              } 
    
    regex = re.compile('file_(.*).csv')
    file_list = os.listdir(fold_directory)
    file_list = [file_name for file_name in file_list if regex.search(file_name)]
    file_list = sorted(file_list)
    k = len(file_list)
    key_hash_list = None
    
    try:
        key_hash_list = switch[k](dataset_name)
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
        

def fold_3_keys(dataset_name='keck'):
    hash_dict = {
    'keck':[('file_0.csv', b'\xc2)4\x85\x8f\xf5\x17Wm\xcf\xbbG\x9a\x81\x03\xa6'),
             ('file_1.csv', b'\xdd<M\xf1\xf43+\xe5\xf7\xa6\xc3:\xf2X5\xca'),
             ('file_2.csv', b'\x93r\xa8\xd0\x8d\xecS\x058j\xc0\x05\x04<\xae\xba')],
    'pcba':[('file_0.csv', b'\xec\x82jO\xfc\xf9\xda\xbcc\x07F\x88\xe3\xe8\xcf\xb6'),
             ('file_1.csv', b'\xa2l\x8e\x9e\xc9\xdb\xe54+*\xc91\xc7j|\x91'),
             ('file_2.csv', b'y:\xa6F\xafL\xf5\xe2q\x13\x14\xea7W\xa0\x9b')],
    'keck_pcba':[('file_0.csv', b'3\x8b\xc9\xc5\xdd\xf0XH\x8b2!\x8d>\xbf\xea\xca'),
                 ('file_1.csv', b'\x1a\xa8\xef\xe3\x11\x14\xcdO\xf9M>\xdc\x89\xaa\x84_'),
                 ('file_2.csv', b'n\x87\xbc\xc3{\xe0\x0e;y=#\x82[t\x9a\xba')]
    }
    return hash_dict[dataset_name]
    
def fold_4_keys(dataset_name='keck'):
    hash_dict = {
    'keck':[('file_0.csv', b'\xd5\x01\x9d\x9eC9\xba\xc1\x08\xb9\x7fjp\x02; '),
             ('file_1.csv', b'\x9e\x04\xce\xca\x92\xc8\xc5J\x81\xb2m\xc6)\x16?\x9a'),
             ('file_2.csv', b'W\xd9\x1c\x16R\x06\xe8\xba\xe9$\xdew\xe3\x01\xe8E'),
             ('file_3.csv', b'q\x02\xee\xc7x,\xb4|"L\x10\x98EI\xd1\xfe')],
    'pcba':[('file_0.csv', b'o\xba>Gi\xfc\xf4lJ\x9b\x1bx\xc10x+'),
             ('file_1.csv', b'">\xe1\xce\xcc\xfa8*#}\x07T\x02b\x87\xc5'),
             ('file_2.csv', b'\xce\xe6\xa3\x9e8Ar\x88c\xfd3\xfaK\xe0\xd1b'),
             ('file_3.csv', b'|\xcc:t\xe4\xcb\x9a\xaf\xeb\xcf\x0e{\x06\xdb\xd8I')],
    'keck_pcba':[('file_0.csv', b'`\xf5N\xce2\rz.\xab}\n\x81sr\x02\x10'),
                 ('file_1.csv', b'\x1d\xfd\x95@\x87\x8c+\x9f\x00\x80\xc3-z<\xabl'),
                 ('file_2.csv', b'u6\xb6\x1b\x7fs\x87}n\xd6h\x1d|g\xc8\x90'),
                 ('file_3.csv', b'\x14\xc2\xb6\x84i\x98g\x98dx\x00\x88\xc3\x0emV')]
    }
    return hash_dict[dataset_name]

def fold_5_keys(dataset_name='keck'):
    hash_dict = {
    'keck':[('file_0.csv', b'\xfb\x97\xd0\xdcs\xd5\x12^\xfc\x96\xef1\\\x03\x0e\n'),
             ('file_1.csv', b'\x11F\xf9\xa3v2\xcaW\xf0\xb5\xc8\x1dZ\xae\x0c\xb6'),
             ('file_2.csv', b'\x05\xfd\t^\x18_M\xcaJ\xc6\x81\xc8PE\xed\x06'),
             ('file_3.csv', b'\xaa0LG\xc3\xe0\xe1y\xf0\xe1G\x84\xc9uz\xed'),
             ('file_4.csv', b'\x94\xf1\xdfw\x81\x89/,\x94\x12\x9f\x9c\xc1\xc2.\xe8')],
    'pcba':[('file_0.csv', b'wh\x81^:{\xcb\x11V\xa8\xfcf\xc0\xdf\x08\x8d'),
             ('file_1.csv', b'p\x99\xcc\xe1D\xc2\xafA\x14\xa2\x00\xc6\xa2\x7f\xc6\xce'),
             ('file_2.csv', b'\xe6\x9f\x8d\x0f\x88\xc6\xb0Fd\xb7\xa4\x92d\nwu'),
             ('file_3.csv', b')\xf1\xd0@\x1f\t\xdd\xa5\x93\xc3\n\x9c&\x90\xe6E'),
             ('file_4.csv', b'\xb8\x93\xe7\x851\xef5\xe9\xc3r`\x95\r\xf5\xc8\xa8')],
    'keck_pcba':[('file_0.csv', b'}\xddy\x8c\xcc"J\\1\xbf\x1b1\x9d\xdf\x9c\x11'),
                 ('file_1.csv', b'\x80\x9b)\x9fz3\x83\x02\r1,D\xadE"\x82'),
                 ('file_2.csv', b'\xd1N\xf8\'\xec\xc6\xbe\x9a"\x7f\x7f$\xec\x12\xacs'),
                 ('file_3.csv', b'\xe3\x1a\xd8\xf2\x9b?\xa1\xdfE\x1c?\x94F0\x8e\xab'),
                 ('file_4.csv', b'\xebB\xaf\x17B\xe5xXp\x16\xf1kh\x82\xd9\x1b')]
    }
    return hash_dict[dataset_name]