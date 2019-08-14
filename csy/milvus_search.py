import sys, getopt
from collections import defaultdict
import pandas as pd
import numpy  as np
import time
import random
import os
from milvus import Milvus, Prepare, IndexType, Status

MILVUS = Milvus()
SERVER_ADDR = "127.0.0.1"
SERVER_PORT = 19530
NQ = 0

CSV = False
# CSV = True
UINT8 = False
# UINT8 = True

# have to set the query's folder name
NQ_FOLDER_NAME = '/home/csy/files/github/random_data/search'
RE_FOLDER_NAME = './search_output'

def connect_server():
    print("connect to milvus.")
    status =  MILVUS.connect(host=SERVER_ADDR, port=SERVER_PORT,timeout = 1000 * 1000 * 20 )
    handle_status(status=status)
    return status


# the status of milvus
def handle_status(status):
    if status.code != Status.SUCCESS:
        print(status)
        sys.exit(2)

def load_nq_vec(NQ):
    filenames = os.listdir(NQ_FOLDER_NAME)
    filenames.sort()
    for filename in filenames:
        filename = NQ_FOLDER_NAME + '/' +filename
        if CSV==True:
            data = pd.read_csv(filename,header=None)
            data = np.array(data)
        else:
            data = np.load(filename)
        if UINT8==True:
            data = (data+0.5)/255
        vec_list = []
        if NQ != 0:
            nb = NQ
        else:
           nb = len(data)
        for i in range(nb):
            vec_list.append(data[i].tolist())
    return vec_list


def save_re_to_file(table_name, rand, results):
    if not os.path.exists(RE_FOLDER_NAME):
        os.mkdir(RE_FOLDER_NAME)
    fname = './'+RE_FOLDER_NAME+'/'+ table_name + '_output.txt'
    with open(fname,'w') as f:
        for i in range(len(results)):
            for j in range(len(results[i])):
                if rand != None:
                    line = str(rand[i]) + ' ' + str(results[i][j].id) + ' ' + str(results[i][j].distance)
                else:
                    line = str(i) + ' ' + str(results[i][j].id) + ' ' + str(results[i][j].distance)
                f.write(line+'\n')
            f.write('\n')
    f.close()
# -s
# search the vectors from milvus and write the results
def search_vec_list(table_name,nq,topk):
    rand = None
    query_list = []
    vectors = load_nq_vec(NQ)
    if nq != 0:
        try:
            rand = sorted(random.sample(range(0,NQ),nq))
            for i in rand:
                query_list.append(vectors[i])
        except:
            print("Error: please change NQ as the num of query list")
            sys.exit()
    else:
        query_list = vectors

    print("table name:", table_name, "query list:", len(query_list), "topk:",topk)
    time_start = time.time()
    status, results = MILVUS.search_vectors(table_name=table_name, query_records=query_list, top_k=topk)
    #status, results = MILVUS.search_vectors_in_files(table_name=table_name, file_ids = file_ids, query_records=query_list, top_k=k)
    time_end = time.time()
    time_cost=time_end - time_start
    print("time_search = ", time_cost)

    time_start = time.time()
    save_re_to_file(table_name, rand, results)
    time_end = time.time()
    time_cost=time_end - time_start
    print("time_save = ", time_cost)


def main():
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "hsq:k:",
            ["help","search","table=","nq=","topk="],
        )
    except getopt.GetoptError:
        print("Usage: test.py --table <table_name> [-q <nq>] -k <topk> -s")
        sys.exit(2)
    nq = 0
    for opt_name, opt_value in opts:
        if opt_name in ("-h", "--help"):
            print("test.py -table <table_name> [-q <nq>] -k <topk> -s")
            sys.exit()
        elif opt_name == "--table":
            table_name = opt_value
        elif opt_name in ("-q", "--nq"):
            nq = int(opt_value)
        elif opt_name in ("-k", "--topk"):
            topk = int(opt_value)
        elif opt_name == "-s":
            connect_server()
            search_vec_list(table_name,nq,topk)    #test.py --table <tablename> -q <nq> -k <topk> [-a] -s
            sys.exit()
if __name__ == '__main__':
    main()
