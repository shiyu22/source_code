import datetime
import time
import os
import sys, getopt
import random
from collections import defaultdict
import pandas as pd
import numpy  as np
from multiprocessing import Process
from functools import reduce
import struct

FOLDER_NAME_BASE ='/home/pensees/feature_compare/card_72_csv'
#FOLDER_NAME_BASE ='/home/pensees/feature_compare/capture_csv_bak'
FOLDER_NAME_CMP ='/home/pensees/feature_compare/capture_csv_bak'
FILE_VET_GT='l2_results0.npy'
ALL = False
# CSV = False
CSV = True
UINT8 = False
# UINT8 = True



def load_vec_csv(file_name):
    if CSV==True:
        data = pd.read_csv(file_name,header=None)
        data = np.array(data)
    else:
        data = np.load(file_name)
    if UINT8==True:
        data = (data+0.5)/255
    vec_list = []
    nb = len(data)
    for i in range(nb):
        vec_list.append(data[i].tolist())
    return vec_list

def calInnerDistance(vec1,vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        dist = np.inner(vec1,vec2)
        return dist

def compare(base,v_cmp):
    base_file = FOLDER_NAME_BASE + '/' + base +".csv"
    print(base_file)
    vec1 = load_vec_csv(base_file)
    if v_cmp!=None:
        v_cmp_file = FOLDER_NAME_CMP + '/' + v_cmp +".csv"
        print(v_cmp_file)
        vec2 = load_vec_csv(v_cmp_file)
        dist = calInnerDistance(vec1,vec2)
        print(dist[0][0])
        return
    data = np.load(FILE_VET_GT)
    vec_list = data.tolist()
    print(len(vec_list))
    for vec2 in vec_list:
        dist = calInnerDistance(vec1,vec2)
        print(dist[0])
    


def main():
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "hb:c:p",
            ["help", "base=", "cmp=", "compare"],
        )
    except getopt.GetoptError:
        print("Usage: test.py -q <nq> -k <topk> -t <table> -l -s")
        sys.exit(2)
    base = None
    v_cmp = None
    for opt_name, opt_value in opts:
        if opt_name in ("-h", "--help"):
            print("test.py -q <nq> -k <topk> -t <table> -l -s")
            sys.exit()
        elif opt_name in ("-b", "--base"):
            base = opt_value
        elif opt_name in ("-c", "--cmp"):
            v_cmp = opt_value
        elif opt_name in ("-p", "--compare"):    #test.py -q <nq> -k <topk> -l
            compare(base,v_cmp)
            sys.exit()
if __name__ == '__main__':
    main()
