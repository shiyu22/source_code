from milvus import Milvus, Prepare, IndexType, Status
import sys, getopt
import numpy  as np
import pandas as pd
import time
import datetime
import random
import os

MILVUS = Milvus()
SERVER_ADDR = "192.168.1.10"
SERVER_PORT = 19530
NQ = 0

CSV = False
# CSV = True
# UINT8 = Falsels
UINT8 = True

NQ_FOLDER_NAME = 'E:/BaiduPan/000_to_299/'
RE_FOLDER_NAME = 'E:/BaiduPan/performance_output'
# nq_scope = [1,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800]
# topk_scope = [1,20,50,100,300,500,800,1000]
nq_scope = [1,10,20,50,100]
topk_scope = [1,10,20,50]


def load_nq_vec(nq):
    filenames = os.listdir(NQ_FOLDER_NAME)
    filenames.sort()
    vectors = []
    length = 0
    flag = 0
    for filename in filenames:
        vec_list = load_vec_list(NQ_FOLDER_NAME + '/' + filename)
        length += len(vec_list)
        if (length > nq):
            flag = 1
            num = length % nq
            if (num != 0 and flag == 0):
                vec_list = load_vec_list(NQ_FOLDER_NAME + '/' + filename, num)
            else:
                vec_list = load_vec_list(NQ_FOLDER_NAME + '/' + filename, nq)
        vectors += vec_list
        if len(vectors) == nq:
            return vectors

def load_vec_list(file_name,num=0):
    if CSV==True:
        data = pd.read_csv(file_name,header=None)
        data = np.array(data)
    else:
        data = np.load(file_name)
    if UINT8==True:
        data = (data+0.5)/255
    vec_list = []
    nb = len(data)
    if(num!=0):
        for i in range(num):
            vec_list.append(data[i].tolist())
        return vec_list
    for i in range(nb):
        vec_list.append(data[i].tolist())
    return vec_list

def connect_server():
    print("connect to milvus")
    status = MILVUS.connect(host=SERVER_ADDR, port=SERVER_PORT, timeout=999 * 1000 * 20)
    handle_status(status=status)
    return status

def handle_status(status):
    if status.code != Status.SUCCESS:
        print(status)
        sys.exit(2)

def search_vec_list(table_name):
    random1 = nowTime = datetime.datetime.now().strftime("%m%d%H%M")
    if not os.path.exists(RE_FOLDER_NAME):
        os.mkdir(RE_FOLDER_NAME)
    filename = RE_FOLDER_NAME + '/' + str(random1) + '_results.csv'
    file = open(filename, "w+")
    file.write('nq,topk,total_time,avg_time' + '\n')
    for nq in nq_scope:
        time_start = time.time()
        query_list = load_nq_vec(nq)
        time_end = time.time()
        print("load query[0]:",len(query_list[0]),"time_load = ",time_end - time_start)
        for k in topk_scope:
            time_start = time.time()
            status, results = MILVUS.search_vectors(table_name=table_name, query_records=query_list, top_k=k)
            time_end = time.time()
            time_cost = time_end - time_start
            line = str(nq) + ',' + str(k) + ',' + str(round(time_cost, 4)) + ',' + str(round(time_cost / nq, 4)) + '\n'
            file.write(line)
            print(nq, k, time_cost)
        file.write('\n')
    file.close()
    print("search_vec_list done !")

def table_show():
    print(MILVUS.show_tables()[1])

def table_describe(table_name):
    print(MILVUS.describe_table(table_name)[1])

def main():
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "hst:",
            ["help", "table=","show","desc"],
        )
    except getopt.GetoptError:
        print("Usage: test.py -q <nq> -k <topk> -t <table> -l -s")
        sys.exit(2)
    for opt_name, opt_value in opts:
        if opt_name in ("-h", "--help"):
            print("test.py -q <nq> -k <topk> -t <table> -l -s")
            sys.exit()
        elif opt_name in ("-t","--table"):
            table_name = opt_value
        elif opt_name == "-s":
            connect_server()
            search_vec_list(table_name)  # test.py --table <tablename> -q <nq> -k <topk> [-a] -s
            sys.exit()
        elif opt_name == "--show":
            connect_server()    #test.py --show
            table_show()
        elif opt_name == "--desc":
            connect_server()    #test.py --show
            table_describe(table_name)
if __name__ == '__main__':
    main()