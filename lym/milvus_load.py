import getopt
import os
import sys
import time
from functools import reduce
import pandas as pd

import numpy as np
import psycopg2
from milvus import *

TO_TXT = True
TO_PG = False
MILVUS_TABLE = 'test1_sq8'
PG_TABLE_NAME = MILVUS_TABLE

FILE_NPY_PATH = '/data/shiyu/data'
FILE_CSV_PATH = '/data/lym/gnoimi/filecsv'
FILE_FVECS_PATH = '/mnt/data/base.fvecs'

FVECS_VEC_NUM = 1000000000
FVECS_BASE_LEN = 100000

SERVER_ADDR = "192.168.1.10"
SERVER_PORT = 19530
milvus = Milvus()
file_index = 0

PG_HOST = "192.168.1.10"
PG_PORT = 5432
PG_USER = "postgres"
PG_PASSWORD = "zilliz123"
PG_DATABASE = "postgres"


def normaliz_data(vec_list):
    for i in range(len(vec_list)):
        vec = vec_list[i]
        square_sum = reduce(lambda x, y: x + y, map(lambda x: x * x, vec))
        sqrt_square_sum = np.sqrt(square_sum)
        coef = 1 / sqrt_square_sum
        vec = list(map(lambda x: x * coef, vec))
        vec_list[i] = vec
    return vec_list


def load_npy_data(filename):
    filename = FILE_NPY_PATH + "/" + filename
    # print(filename)
    data = np.load(filename)
    data = data.tolist()
    # data = normaliz_data(data)
    return data


def load_csv_data(filename):
    filename = FILE_CSV_PATH + "/" + filename
    # print(filename)
    data = pd.read_csv(filename, header=None)
    data = np.array(data)
    data = data.tolist()
    # data = normaliz_data(data)
    return data


def load_fvecs_data(fname, base_len, idx):
    begin_num = base_len * idx
    # print(fname, ": ", begin_num )
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    data = x.view('float32').reshape(-1, d + 1)[begin_num:(begin_num + base_len), 1:]
    data = data.tolist()
    # data = normaliz_data(data)
    return data


def handle_status(status):
    if status.code != Status.SUCCESS:
        print(status)
        sys.exit(2)


def connect_milvus_server():
    print("connect to milvus")
    status = milvus.connect(host=SERVER_ADDR, port=SERVER_PORT, timeout=1000 * 1000 * 20)
    handle_status(status=status)
    return status


def connect_postgres_server():
    try:
        conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD,
                                database=PG_DATABASE)
        print("connect the database!")
        return conn
    except:
        print("unable to connect to the database")


def create_pg_table(conn, cur):
    try:
        sql = "CREATE TABLE " + PG_TABLE_NAME + " (ids bigint,  vecs float[]);"
        cur.execute(sql)
        conn.commit()
        print("create postgres table!")
    except:
        print("can't create postgres table")
    try:
        sql = "alter table " + PG_TABLE_NAME + " alter column vecs set storage EXTENDED;"
        cur.execute(sql)
        conn.commit()
        print("toast success")
    except:
        print("faild toast pg table")


def create_pg_vecs_table(conn, cur):
    try:
        sql = "CREATE TABLE " + PG_TABLE_NAME + " (ids bigint,  location bigint);"
        cur.execute(sql)
        conn.commit()
        print("create postgres table!")
    except:
        print("can't create postgres table")


def insert_data_to_pg(ids, vector, conn, cur):
    try:
        sql = "INSERT INTO " + PG_TABLE_NAME + " VALUES(" + str(ids) + ", array" + str(vector) + ");"
        cur.execute(sql)
        conn.commit()
        print("insert success!")
    except:
        print("faild insert")


def copy_data_to_pg(conn, cur):
    # fname = './temp.csv'
    sql = "copy " + PG_TABLE_NAME + " from " + "'/root/pg_map.csv'" + " with CSV delimiter '|';"
    # print(sql)
    try:
        cur.execute(sql)
        conn.commit()
        print("copy data to pg sucessful!")
    except:
        print("faild  copy!")


def create_pg_index(conn, cur):
    try:
        sql = "CREATE INDEX index_ids on " + PG_TABLE_NAME + "(ids);"
        cur.execute(sql)
        conn.commit()
        print("build index sucessful!")
    except:
        print("faild build index")


def record_id_map(ids, table_name):
    global file_index
    filename = './' + 'idmap/' + table_name + '_idmap.txt'
    with open(filename, 'a') as f:
        for i in range(len(ids)):
            line = str(ids[i]) + " %03d%06d\n" % (file_index, i)
            f.write(line)
    file_index += 1


def record_vecs_id_map(ids, count, table_name):
    filename = './' + 'idmap/' + table_name + '_idmap.txt'
    with open(filename, 'a') as f:
        for i in range(len(ids)):
            location_index = count * FVECS_BASE_LEN + i
            line = str(ids[i]) + " " + str(location_index) + "\n"
            # print(line)
            f.write(line)


def record_vec(ids, vectors):
    fname = 'temp.csv'
    with open(fname, 'w+') as f:
        for i in range(len(ids)):
            line = str(ids[i]) + "|{" + str(vectors[i]).strip('[').strip(']') + "}\n"
            f.write(line)


def record_fvecs_csv(ids, base_len, count):
    fname = 'pg_map.csv'
    with open(fname, 'w+') as f:
        for i in range(len(ids)):
            location = count * base_len + i
            line = str(ids[i]) + "|" + str(location) + "\n"
            f.write(line)


def main(argv):
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "ncfb",
            ["npy", "csv", "fvecs", "bvecs"],
        )
        # print(opts)
    except getopt.GetoptError:
        print("Usage: load_vec_to_milvus.py -n <npy>  -c <csv> -f <fvecs> -b <bvecs>")
        sys.exit(2)

    for opt_name, opt_value in opts:
        if opt_name in ("-n", "--npy"):
            connect_milvus_server()
            if TO_PG:
                conn = connect_postgres_server()
                cur = conn.cursor()
                create_pg_table(conn, cur)
            filenames = os.listdir(FILE_NPY_PATH)
            filenames.sort()
            for filename in filenames:
                print(filename)
                vectors = load_npy_data(filename)
                print(len(vectors))
                time_add_start = time.time()
                status, ids = milvus.add_vectors(table_name=MILVUS_TABLE, records=vectors)
                time_add_end = time.time()
                print("insert milvus time: ", time_add_end - time_add_start)
                if TO_PG:
                    i = 0
                    time_pg_strat = time.time()
                    for id in ids:
                        insert_data_to_pg(id, vectors[i], conn, cur)
                        i = i + 1
                    time_pg_end = time.time()
                    print("import to pg time: ", time_pg_end - time_pg_strat)
                if TO_TXT:
                    try:
                        os.mkdir('./idmap')
                    except:
                        idmap = "True"
                    record_id_map(ids, MILVUS_TABLE)
            if TO_PG:
                create_pg_index(conn, cur)

        elif opt_name in ("-c", "--csv"):
            connect_milvus_server()
            if TO_PG:
                conn = connect_postgres_server()
                cur = conn.cursor()
                create_pg_table(conn, cur)
            filenames = os.listdir(FILE_CSV_PATH)
            filenames.sort()
            for filename in filenames:
                print(filename)
                vectors = load_csv_data(filename)
                print(len(vectors))
                time_add_start = time.time()
                status, ids = milvus.add_vectors(table_name=MILVUS_TABLE, records=vectors)
                time_add_end = time.time()
                print("insert time: ", time_add_end - time_add_start)

                if TO_PG:
                    i = 0
                    time_pg_start = time.time()
                    for id in ids:
                        insert_data_to_pg(id, vectors[i], conn, cur)
                        i = i + 1
                    time_pg_end = time.time()
                    print("import to pg time: ", time_pg_end - time_pg_start)
                if TO_TXT:
                    try:
                        os.mkdir('./idmap')
                    except:
                        idmap = "True"
                    record_id_map(ids, MILVUS_TABLE)
            if TO_PG:
                create_pg_index(conn, cur)

        elif opt_name in ("-f", "--fvecs"):
            connect_milvus_server()

            if TO_PG:
                conn = connect_postgres_server()
                cur = conn.cursor()
                create_pg_vecs_table(conn, cur)

            count = 0
            while count < (FVECS_VEC_NUM // FVECS_BASE_LEN):
                vectors = load_fvecs_data(FILE_FVECS_PATH, FVECS_BASE_LEN, count)
                print(len(vectors))
                time_add_start = time.time()
                status, ids = milvus.add_vectors(table_name=MILVUS_TABLE, records=vectors)
                time_add_end = time.time()
                print(count, " insert to milvus time: ", time_add_end - time_add_start)
                if TO_PG:
                    i = 0
                    time_pg_start = time.time()
                    record_fvecs_csv(ids, FVECS_BASE_LEN, count)
                    copy_data_to_pg(conn, cur)
                    time_pg_end = time.time()
                    print("copy data to pg time: ", time_pg_end - time_pg_start)
                if TO_TXT:
                    try:
                        os.mkdir('./idmap')
                    except:
                        idmap = "True"
                    record_vecs_id_map(ids, count, MILVUS_TABLE)
                    # print(len(ids))
                count = count + 1
            if TO_PG:
                create_pg_index(conn, cur)


        else:
            print("wrong parameter")
            sys.exit(2)


if __name__ == "__main__":
    main(sys.argv[1:])
