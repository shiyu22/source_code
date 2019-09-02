import datetime
import getopt
import os
import random
import sys
import time
from functools import reduce
from multiprocessing import Process
import numpy  as np
from milvus import Milvus, IndexType, Status

MILVUS = Milvus()
SERVER_ADDR = "0.0.0.0"
SERVER_PORT = 19530
nq_scope = [2, 4, 6, 8, 10]
topk_scope = [1, 10, 100]


# the status of milvus
def handle_status(status):
    if status.code != Status.SUCCESS:
        print( status )
        sys.exit( 2 )


# connect to the milvus server
def connect_server():
    print( "connect to milvus" )
    status = MILVUS.connect( host=SERVER_ADDR, port=SERVER_PORT, timeout=1000 * 1000 * 20 )
    handle_status( status=status )
    return status


# -c
# create the table with milvus
def create_table(table_name, dim, index_type):
    if index_type == 'flat':
        it = IndexType.FLAT
    elif index_type == 'ivf':
        it = IndexType.IVFLAT
    elif index_type == 'ivfsq8':
        it = IndexType.IVF_SQ8
    param = {'table_name': table_name, 'dimension': dim, 'index_type': it, 'store_raw_vector': False}
    print( "create table: ", table_name, " dimension:", dim, " index_type:", it )
    return MILVUS.create_table( param )


# --show
def table_show():
    print( MILVUS.show_tables()[1] )


# --describe
def describe_table(table_name):
    print( MILVUS.describe_table( table_name )[1] )


# -d
def delete_table(table_name):
    MILVUS.delete_table( table_name=table_name )


# --build
def build_table(table_name):
    MILVUS.build_index( table_name )

def show_server_version():
    print(MILVUS.server_version()[1])

def show_client_version():
    print(MILVUS.client_version())



# -s
# search the vectors from milvus and write the results
# def search_vec_list(table_name):
#     connect_server()
#     query_list = []
#     for i in nq_scope:
#         arr = gt[:] # 将groundtruth的列表复制到arr
#         np.random.shuffle(arr)
#         for j in nq_scope[i]:
#             query_list.append(arr[j])
#         for k in topk_scope:
#             time_start = time.time()
#             status, results = MILVUS.search_vectors(table_name=table_name, query_records=query_list, top_k=k)
#             time_end = time.time()
#             print("time=", time_end - time_start)
#             save_id_to_file(results, table_name)

# get list[ids]=map
def get_id_map(table_name):
    filename = table_name + "_idmap.txt"
    res = dict()
    for line in open( filename, 'r' ):
        key_val = line.split()
        key = key_val[0]
        val = key_val[1]
        res[key] = val
    return res


# save date(ids,maps)
def save_id_to_file(results, table_name):
    filename = table_name + '_output.txt'
    dic = get_id_map( table_name )

    with open( filename, 'w' ) as f:
        for r in results:
            for score in r:
                index = dic.get( str( score.id ) )
                if index != None:
                    f.write( index + '\n' )
            f.write( '\n' )


def main():
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "chd:",
            ["help", "table=", "dim=", "index=", "nq=", "show", "describe", "delete", "build", "server_version",
             "client_version"]
        )
    except getopt.GetoptError:
        print( "Usage: python milvus_toolkit.py -q <nq> -k <topk> -c <table> -s" )
        sys.exit( 2 )

    for opt_name, opt_value in opts:
        if opt_name in ("-h", "--help"):
            print( "python milvus_toolkit.py test.py -q <nq> -k <topk> -c <table> -c -s" )
            sys.exit()
        elif opt_name == "--table":
            table_name = opt_value
        elif opt_name == "--dim":
            dim = int( opt_value )
        elif opt_name == "--index":
            index_type = opt_value
        elif opt_name == "-c":
            connect_server()
            create_table( table_name, dim, index_type )
        elif opt_name == "--show":
            connect_server()
            table_show()
        elif opt_name == "--describe":
            connect_server()
            describe_table( table_name )
        elif opt_name in ("-d", "--delete"):
            connect_server()
            delete_table( table_name )
        elif opt_name == "--build":
            connect_server()
            build_table( table_name )
        elif opt_name == "--server_version":
            connect_server()
            show_server_version()
        elif opt_name == "--client_version":
            connect_server()
            show_client_version()
        # elif opt_name == "-s":
        #     search_vec_list(table_name, nq, topk)
        #     sys.exit()


if __name__ == '__main__':
    main()
