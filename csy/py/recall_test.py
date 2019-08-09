import datetime
import time
import os
import sys, getopt
import random
from collections import defaultdict
import pandas as pd
import numpy  as np
from milvus import Milvus, Prepare, IndexType, Status
from multiprocessing import Process
from functools import reduce
import struct

MILVUS = Milvus()
SERVER_ADDR = "127.0.0.1"
SERVER_PORT = 19530
TABLE_DIMENSION = 512
FILE_PREFIX = "binary_"
INSERT_BATCH = 10000
FILE_GT = 'ground_truth_all'
FILE_GT_T = 'ground_truth.txt'
file_index = 0
#FOLDER_NAME = './random_data'
FILE_NAME ='/home/pensees/feature_compare/capture_csv'
NQ = 0
TOPK = 0
ALL = False
# CSV = False
CSV = True
UINT8 = False
# UINT8 = True

# FILE_START = 1960
# FILE_NUM = 300

nq_scope = [1]
topk_scope = [1]



def connect_server():
    print("connect to milvus")
    status =  MILVUS.connect(host=SERVER_ADDR, port=SERVER_PORT,timeout = 1000 * 1000 * 20 )
    # handle_status(status=status)
    return status


def load_vec_list_from_file(file_name, nb = 0):
    # import numpy as np
    # data = np.load(file_name)
    if CSV==True:
        data = pd.read_csv(file_name,header=None)
        data = np.array(data)
    else:
        data = np.load(file_name)
    if UINT8==True:
        data = (data+0.5)/255
    vec_list = []
    if nb == 0:
        nb = len(data)
    for i in range(nb):
        vec_list.append(data[i].tolist())
    return vec_list

def compare(table_name,time,results,num,nq,topk):
    filename = table_name + '_'+ str(num+1)+ '_result_all.csv'
    # num=[]
    # for line in open(filename):
    #     if line != "\n":
    #         line=line.strip()
    #         num.append(line)
    re_list = []
    com_list=[]
    for line in open('ground_truth.txt'):
        if line != "\n":
            line=line.strip()
            com_list.append(line)
            # print(line)
    with open(filename,'w') as f:
        f.write('topk,基准ID,搜索结果,distance,recall' + '\n')
        i=0
       # recall = 'N'
        while i<nq:
            j=0
            while j<topk:
                recall = 'N'
                if str(com_list[num])==str(results[i][j].id):
                    recall = "100%"
                # print(com_list[num])
                line = str(topk) + ',' + com_list[num]+ ',' + str(results[i][j].id) + ',' +str(round(results[i][j].distance,3)) + ',' + recall
                f.write(line+'\n')
                j=j+1
            i=i+1
            f.write('\n')
        f.write('time='+str(time))
    f.close


def search_vec_list(table_name,name,nq=0,topk=0,nb=0,num=0):
    connect_server()
    query_list = []
    file_name = FILE_NAME+'/'+name
    print(file_name)
    filenames = os.listdir(file_name)  # 得到文件夹下的所有文件名称
    filenames.sort()
    if nq==0:
        nq = len(filenames)
    print("nq:",nq)
    nq_0 = nq
    for filename in filenames:
        # vec_list = load_vec_list_from_file(FOLDER_NAME+'/'+filename)
        if(nq_0<=0):
            break
        nq_0 -= 1
        filename = file_name+'/'+filename
        print(filename)
        query_vec = load_vec_list_from_file(file_name = filename,nb = 0)
        query_list += query_vec
    # print(query_list)
    print("query list :",len(query_list))
    time_start = time.time()
    status, results = MILVUS.search_vectors(table_name=table_name, query_records=query_list, top_k=topk)
    #status, results = MILVUS.search_vectors_in_files(table_name=table_name, file_ids = file_ids, query_records=query_list, top_k=k)
    time_end = time.time()
    time_cost_s=time_end - time_start
    print("time_search=", time_cost_s)
    # print('distance=',results[0][0].distance)
    time_start = time.time()
    compare(table_name,time_cost_s,results,num,nq,topk)
    time_end = time.time()
    time_cost=time_end - time_start
    print("time_compare=", time_cost)





def main():
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "hsn:m:q:k:",
            ["help", "search","nb=","table=","num=","nq=","topk=","test="],
        )
    except getopt.GetoptError:
        print("Usage: test.py -q <nq> -k <topk> -t <table> -l -s")
        sys.exit(2)
    nb = 0
    nq = 0
    for opt_name, opt_value in opts:
        if opt_name in ("-h", "--help"):
            print("test.py -q <nq> -k <topk> -t <table> -l -s")
            sys.exit()
        elif opt_name == "--table":
            table_name = opt_value
        elif opt_name in ("-q", "--nq"):
            nq = int(opt_value)
        elif opt_name in ("-k", "--topk"):
            topk = int(opt_value)
        elif opt_name in ("-n", "--nb"):
            nb = int(opt_value)
        elif opt_name in ("--test"):
            name = opt_value
        elif opt_name in ("-m", "--num"):
            num = int(opt_value)
        elif opt_name == "-s":
            # print(table_name,nq,topk,nb,num)
            search_vec_list(table_name,name,nq,topk,nb,num)    #test.py --table <tablename> -q <nq> -k <topk> [-a] -s
            sys.exit()

if __name__ == '__main__':
    main()

