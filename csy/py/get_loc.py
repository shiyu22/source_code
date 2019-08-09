import fire
import pandas as pd
import numpy as np
import os

FOLDER_NAME = '/home/pensees/feature_float'
STR = '000000000'

def fun(STR):
    STR = str(STR)
    loca = int(STR[0:3])
    offset = int(STR[3:9])
    print(loca,offset)
    get_info(loca,offset)

def get_info(loca,offset):
    filenames = os.listdir(FOLDER_NAME)
    filenames.sort()
    print("filename:",filenames[loca])
    print("offset:",offset+1)
    fname = FOLDER_NAME + '/' + filenames[loca]
    data = pd.read_csv(fname,header=None)
    data = np.array(data)
    data = data.tolist()
    print(data[offset])

if __name__ == '__main__':
    fire.Fire(fun)
