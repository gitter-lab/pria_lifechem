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


def generate_(fold_directory):
    regex = re.compile('file_(.*).csv')
    file_list = os.listdir(fold_directory)
    file_list = [file_name for file_name in file_list if regex.search(file_name)]
    file_list = sorted(file_list)
    md5_hash_list = [(fname, hashlib.md5(open(fold_directory+fname, 'rb').read()).digest()) for fname in file_list]

    return md5_hash_list


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

    md5_hash_list = generate_(fold_directory)
    
    no_errors = True
    for i in range(k):
        if key_hash_list[i] != md5_hash_list[i]:
            no_errors = False
            break
          
    if no_errors:
        print("All files correct with no errors")
    else:
        print("There are errors in the files")
        

def fold_3_keys(dataset_name='keck'):
    hash_dict = {
    'keck': [('file_0.csv', b'\xc2)4\x85\x8f\xf5\x17Wm\xcf\xbbG\x9a\x81\x03\xa6'),
             ('file_1.csv', b'\xdd<M\xf1\xf43+\xe5\xf7\xa6\xc3:\xf2X5\xca'),
             ('file_2.csv', b'\x93r\xa8\xd0\x8d\xecS\x058j\xc0\x05\x04<\xae\xba')],
    'keck_extended': [('file_0.csv', '\xff\xde\xed/\xe9N\xe1f\x04\xf7\x82\xa7\xe6\xef\x172'),
                      ('file_1.csv', '\x96\x82\xe5J\x10\xdd:y\xc5\x18\x0b\x94\xbfTvg'),
                      ('file_2.csv', '\xe4\x8f\'3\x97Q"Lx!\'i&\xdc\xda\x13')],
    'pcba': [('file_0.csv', '\\2\xfd+\x1d<;\x82\xf6z\xf8h(\x98\x7f}'),
             ('file_1.csv', '\x14\xeb\xa4\x91\x9b\xc1\x11\xd56\x15\x08r\xe1O\x15\xa9'),
             ('file_2.csv', '\xac\xac\x86\x83\x17\x1a\xff\x10Q\xad\xef\xb72\xe1\xf0F')],
    'keck_pcba': [('file_0.csv', '\\2\xfd+\x1d<;\x82\xf6z\xf8h(\x98\x7f}'),
                  ('file_1.csv', '\x14\xeb\xa4\x91\x9b\xc1\x11\xd56\x15\x08r\xe1O\x15\xa9'),
                  ('file_2.csv', '\xac\xac\x86\x83\x17\x1a\xff\x10Q\xad\xef\xb72\xe1\xf0F')]
    }
    return hash_dict[dataset_name]
    
def fold_4_keys(dataset_name='keck'):
    hash_dict = {
    'keck':[('file_0.csv', b'\xd5\x01\x9d\x9eC9\xba\xc1\x08\xb9\x7fjp\x02; '),
             ('file_1.csv', b'\x9e\x04\xce\xca\x92\xc8\xc5J\x81\xb2m\xc6)\x16?\x9a'),
             ('file_2.csv', b'W\xd9\x1c\x16R\x06\xe8\xba\xe9$\xdew\xe3\x01\xe8E'),
             ('file_3.csv', b'q\x02\xee\xc7x,\xb4|"L\x10\x98EI\xd1\xfe')],
    'keck_extended': [('file_0.csv', 'g\xffof:K4\xb9\x1eU\x08\x1b\x93Z\xfb\x81'),
                      ('file_1.csv', '3\xaa\xf3X\xc7\xd7?\xee9\x9a\xb9\x83x\xab\x1b\x80'),
                      ('file_2.csv', '/\xc0\xa5,0\xbe(\x8aL\xde\xc1\x06\x9bu\x9a\x86'),
                      ('file_3.csv', '4\xd5\x8f\xaeS\x97ri\xb2E\xe6)\x00d\xa4\x05')],
    'pcba':[('file_0.csv', '\x00\xc8\x1a\xdc\xc1\x8e@\xb0\xb4w\xa7\xb2^8\xf1\xd5'),
            ('file_1.csv', '=\x95\x92R\x95\x8b\xd6\xf3\xe7qb}T\xadU\x08'),
            ('file_2.csv', 'f>\x99S.\xe0\xe2*Gb\xf9\x98\x97\xaf\x19\xd0'),
            ('file_3.csv', '\xd0\xee\x18iB\xcdi\x85\x0eN\xf5F/\n\xfe\xb5')],
    'keck_pcba':[('file_0.csv', '(\x1f<\xb7\xef\xd5\xd7\xa4m\tq\x0e\xae\x10\xe1\x80'),
                 ('file_1.csv', 'x&\x80\x0c\xa7\xc8PI\xa6v\xd4\xe0/\xd2\xa1g'),
                 ('file_2.csv', '\'Q\xd7;F\x1d\xf3\xa8V\xbc\x81\xa9#"\x0f<'),
                 ('file_3.csv', '\xcff\xe3d5\xa29&v|\xbd\x0c\xb3#\xb0\xea')]
    }
    return hash_dict[dataset_name]

def fold_5_keys(dataset_name='keck'):
    hash_dict = {
    'keck':[('file_0.csv', b'\xfb\x97\xd0\xdcs\xd5\x12^\xfc\x96\xef1\\\x03\x0e\n'),
             ('file_1.csv', b'\x11F\xf9\xa3v2\xcaW\xf0\xb5\xc8\x1dZ\xae\x0c\xb6'),
             ('file_2.csv', b'\x05\xfd\t^\x18_M\xcaJ\xc6\x81\xc8PE\xed\x06'),
             ('file_3.csv', b'\xaa0LG\xc3\xe0\xe1y\xf0\xe1G\x84\xc9uz\xed'),
             ('file_4.csv', b'\x94\xf1\xdfw\x81\x89/,\x94\x12\x9f\x9c\xc1\xc2.\xe8')],
    'keck_extended': [('file_0.csv', '-\xb3\xad\x1e0\x16Z\xaadv\x05h\x03\xa5\x8eh'),
                      ('file_1.csv', '\xe8\x82\xc1\xb8\x04\x15\x05\\\x8dK\xb1x\x18\xf7\x02\x02'),
                      ('file_2.csv', '\xb0\xff\x86\xa9\xb3{\x88\tz\xed\xe6\xc0\xbb\x02\x1f\xd6'),
                      ('file_3.csv', '\x8fF\xed\x97\xf0\x9f\xb0t\x19\x0e\xb5\x1c\xb0^\xaa\x1b'),
                      ('file_4.csv', 'R.\xb2\x0c\x00a\xd6\x12\xfd\x19\n\x10\x87g\x9c\x11')],
    'pcba':[('file_0.csv', '@\xb8\x1c\x0c\xce\xfa\tn\x91\x9a\xf6?S\x9e\x1b\x8d'),
            ('file_1.csv', '!0\xe0>\xf2\xd3\xe8)T\xa9N\xacq\x94\xae\xec'),
            ('file_2.csv', '\x0f\xbe\xce)2\xbf\xac\x03f$\xddK\xa0\x91\x88E'),
            ('file_3.csv', '\x13W\x95K\x0cw\xfe4l~)\xbc\x16\x92\xf0\x91'),
            ('file_4.csv', '\x1b\x19,0 C\xdbu\xb8/\x81l\xc5\n\x7f\x19')],
    'keck_pcba':[('file_0.csv', "\xdd\x16<:Q'Wl7S\xbc\x8e\x83\xf4\xc0\xac"),
                 ('file_1.csv', 'UJ7N\x19\x19\x8a\xc5\xca\xf8\x1c?\x13\xa2\x07\x83'),
                 ('file_2.csv', '\x88um\x97\xef\xe4O\xb8d9\x1eL\xe1\x87[+'),
                 ('file_3.csv', ' \xe6\xa9\x92I\xfe\xccR\x1d\xd7\x8cH\xcc\x99\xa6-'),
                 ('file_4.csv', ')\xb9\x1e\xbdHY\xfe\x93\xdf\xde\xbfT\x13\xa2<e')]
    }
    return hash_dict[dataset_name]


if __name__ == '__main__':
    print 'Check Keck Dataset'
    check_keck_fold_integrity('../dataset/fixed_dataset/fold_3/')
    check_keck_fold_integrity('../dataset/fixed_dataset/fold_4/')
    check_keck_fold_integrity('../dataset/fixed_dataset/fold_5/')
    print

    print 'Check Keck_extended Dataset'
    check_keck_fold_integrity('../dataset/keck_extended/fold_3/', dataset_name='keck_extended')
    check_keck_fold_integrity('../dataset/keck_extended/fold_4/', dataset_name='keck_extended')
    check_keck_fold_integrity('../dataset/keck_extended/fold_5/', dataset_name='keck_extended')
    print
    
    print 'Check PCBA Dataset'
    check_keck_fold_integrity('../dataset/pcba/fold_3/', dataset_name='pcba')
    check_keck_fold_integrity('../dataset/pcba/fold_4/', dataset_name='pcba')
    check_keck_fold_integrity('../dataset/pcba/fold_5/', dataset_name='pcba')
    print
    
    print 'Check Keck_PCBA Dataset'
    check_keck_fold_integrity('../dataset/keck_pcba/fold_3/', dataset_name='keck_pcba')
    check_keck_fold_integrity('../dataset/keck_pcba/fold_4/', dataset_name='keck_pcba')
    check_keck_fold_integrity('../dataset/keck_pcba/fold_5/', dataset_name='keck_pcba')