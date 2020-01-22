import getopt
import os
import sys
import psycopg2
import time
from functools import reduce
import numpy as np
from milvus import *

SERVER_ADDR = "192.168.1.58"
SERVER_PORT = 19550

FILE_UINT8_PATH = '/mnt/workspace/data/pub_0_99/can_smiles_uint8'
FILE_HEX_PATH = '/data/workspace/apptec/1B_data/out_test/out_npy'
# FILE_HEX_PATH = '/data/workspace/apptec/1B_data/out_1B/out_npy'
FILE_IDS_SMILES = '/data/workspace/apptec/1B_data/out_test/test_ids_smiles.csv'

# FILE_SMILES = '/mnt/workspace/data/pub_0_99/can_smiles_smile'
FILE_SMILES = '/data/workspace/apptec/1B_data/out_test/out_smiles'

TO_PG = False
PG_HOST = "192.168.1.58"
PG_PORT = 5432
PG_USER = "zilliz_support"
PG_PASSWORD = "zilliz123"
PG_DATABASE = "demo1b"

milvus = Milvus()
is_uint8 = False


def load_uint8_data(filename):
    filename = FILE_UINT8_PATH + "/" + filename
    print("uint8_file:",filename)
    data = []
    for line in open(filename, 'r'):
        line = line.strip('\n')
        data_uint8 = line.split(' ')
        data_uint8 = list(map(int, data_uint8))
        data_bytes = bytes(data_uint8)
        # print(data_uint8,data_bytes,'\n')
        data.append(data_bytes)
    return data


def handle_status(status):
    if status.code != Status.SUCCESS:
        print(status)
        sys.exit(2)


def connect_milvus_server():
    print("connect to milvus")
    status = milvus.connect(host=SERVER_ADDR, port=SERVER_PORT, timeout=100)
    handle_status(status=status)
    return status


def uint8_to_milvus(MILVUS_TABLE):
    filenames = os.listdir(FILE_UINT8_PATH)
    filenames.sort()

    filenames_smiles = os.listdir(FILE_SMILES)
    filenames_smiles.sort()

    count = 0
    cache = Cache('.')
    cache.reset('size_limit', 21474836480)
    for filename in filenames:
        names = load_smiles(filenames_smiles[count])
        vectors = load_uint8_data(filename)
        total_ids = []
        ids_lens = 0
        print("names:", len(names), "vectors:", len(vectors))
        while ids_lens<len(vectors) :
            time_add_start = time.time()
            # print(vectors_ids)
            try:
                status, ids = milvus.add_vectors(table_name=MILVUS_TABLE, records=vectors[ids_lens:ids_lens+100000])
            except:
                status, ids = milvus.add_vectors(table_name=MILVUS_TABLE, records=vectors[ids_lens:len(vectors)])
            # print(status)
            total_ids += ids
            time_add_end = time.time()
            # print("ids:",len(ids),ids_vec[0])
            print("insert milvus time: ", time_add_end - time_add_start, '\n')
            ids_lens += 100000
            # file_index = file_index + 1

        for i in range(len(names)):
            cache[total_ids[i]] = names[i]
        print("-----------------------------cahce:",len(cache))

        count += 1


def hex_to_milvus(MILVUS_TABLE):
    filenames = os.listdir(FILE_HEX_PATH)
    filenames.sort()

    for filename in filenames:
        vectors = load_hex(filename)
        vectors_ids = []
        for i in range(len(vectors)):
            location = '8' + '%04d'%count  + '%06d'%i
            vectors_ids.append(int(location))

        print("\nvectors_ids:", len(vectors_ids), "vectors:", len(vectors))
        time1 = time.time()
        status, ids = milvus.add_vectors(table_name=MILVUS_TABLE, records=vectors, ids=vectors_ids)
        time2 = time.time()
        print(status,'----------',time2-time1)
        count += 1


def load_smiles(file):
    file = FILE_SMILES + '/' + file
    print("smiles_file:",file)
    smiles = []
    for line in open(file, 'r'):
        data = line.strip('\n')
        # smiles.append(data.encode())
        smiles.append(data)
    return smiles


def load_hex(file):
    file = FILE_HEX_PATH + '/' + file
    print("hex_file:",file)
    data = np.load(file)
    data = data.tolist()
    vectors = []
    for d in data:
        vectors.append(bytes.fromhex(d))
    return vectors


def main(argv):
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "t:uh",
            ["uint8", "hex","table="],
        )
        # print(opts)
    except getopt.GetoptError:
        print("Usage: load_vec_to_milvus.py -n <npy>  -c <csv> -f <fvecs> -b <bvecs>")
        sys.exit(2)

    for opt_name, opt_value in opts:
        if opt_name in ("-t", "--table"):
            MILVUS_TABLE = opt_value
            PG_TABLE_NAME = opt_value
        elif opt_name in ("-u", "--u8"):
            connect_milvus_server()
            uint8_to_milvus(MILVUS_TABLE)
        elif opt_name in ("-h", "--hex"):
            connect_milvus_server()
            hex_to_milvus(MILVUS_TABLE)
        else:
            print("wrong parameter")
            sys.exit(2)

if __name__ == "__main__":
    main(sys.argv[1:])