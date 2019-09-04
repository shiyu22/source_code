import getopt
import sys
import psycopg2
import pandas as pd
import time
import numpy as np
import os

PG_HOST = "192.168.1.127"
PG_PORT = 5432
PG_USER = "postgres"
PG_PASSWORD = "postgres"
PG_DATABASE = "postgres"

PG_VEC = False
PG_LOC = True

UINT8 = True
GT_TOPK = 20

BASE_FOLDER_NAME = '/data/workspace/data/data_2'
GT_FOLDER_NAME = 'ground_truth'
SE_FOLDER_NAME = 'search'
SE_CM_FILE_NAME = '_file_output.txt'
CM_FOLDER_NAME = 'compare'
IDMAP_FOLDER_NAME = 'idmap'
IDMAP_NAME = '_idmap.txt'

GT_NAME = 'location.txt'
GT_FILE_NAME = 'file_location.txt'
GT_VEC_NAME = 'vectors.npy'

SE_FILE_NAME = '_output.txt'
CM_CSV_NAME = '_output.csv'
CM_GET_LOC_NAME = '_loc_compare.txt'


def load_search_out(table_name, ids=[], rand=[], distance=[]):
    file_name = SE_FOLDER_NAME + '/' + table_name + SE_FILE_NAME
    top_k = 0
    with open(file_name, 'r') as f:
        for line in f.readlines():
            data = line.split()
            if data:
                rand.append(data[0])
                ids.append(data[1])
                distance.append(data[2])
            else:
                top_k += 1
    return rand, ids, distance, top_k


def load_gt_out(table_name, flag):
    if flag:
        file_name = GT_FOLDER_NAME + '/' + GT_NAME
    else:
        file_name = CM_FOLDER_NAME + '/' + table_name + CM_GET_LOC_NAME
    loc = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            data = line.split()
            if data:
                loc.append(data)
    return loc


def connect_postgres_server():
    try:
        conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, database=PG_DATABASE)
        print("connect the database!")
        return conn
    except:
        print("unable to connect to the database")


def search_vec_in_pg(table_name, cur, ids):
    try:
        sql = "select vecs from " + table_name + " where ids = " + ids + ";"
        cur.execute(sql)
        rows = cur.fetchall()
        return rows
    except:
        print("search faild!")


def search_loc_in_pg(table_name, cur, ids):
    try:
        sql = "select location from " + table_name + " where ids=" + str(ids) + ";"
        cur.execute(sql)
        rows = cur.fetchall()
        location = str(rows[0][0])
        return location
    except:
        print("search faild!")
        sys.exit()


def load_vec(filename):
    data = np.load(filename)
    if UINT8:
        data = (data + 0.5) / 255
    vec_list = []
    nb = len(data)
    for i in range(nb):
        vec_list.append(data[i].tolist())
    return vec_list


def get_vec_recalls(vec1, vec2, nq, topk, rand):
    count_all = 0
    recalls = []
    for i in range(nq):
        count = 0
        v1 = sorted(vec1[i * topk:i * topk + topk])
        n = int(rand[i * topk])
        v2 = sorted(vec2[n * GT_TOPK:n * GT_TOPK + topk])
        com = (np.round(np.array(v1), 13) == np.round(np.array(v2), 13))
        for c in com:
            if c.all():
                count += 1
        recall = count / topk
        count_all += count
        recalls.append(recall)
    return recalls, count_all


def compare_pg_vec(table_name):
    rand, ids, dis, nq = load_search_out(table_name)
    top_k = int(len(rand) / nq)
    print("nq:", nq, "top_k:", top_k)
    vec1 = []
    conn = connect_postgres_server()
    cur = conn.cursor()
    time1_pg = time.time()
    for i in ids:
        vectors = search_vec_in_pg(table_name, cur, i)
        vec1.append(vectors[0][0])
    time2_pg = time.time()
    print("time cost in pg search:", round(time2_pg - time1_pg, 4))
    time1_com = time.time()
    vec2 = np.load(GT_FOLDER_NAME + '/' + GT_VEC_NAME)
    vec2 = vec2.tolist()
    recalls, count_all = get_vec_recalls(vec1, vec2, nq, top_k, rand)
    time2_com = time.time()
    print("time cost with compare files:", round(time2_com - time1_com, 4))
    save_compare_csv(nq, top_k, recalls, count_all, table_name)


def save_compare_csv(nq, top_k, recalls, count_all, table_name):
    with open(CM_FOLDER_NAME + '/' + table_name + '_' + str(nq) + "_" + str(top_k) + CM_CSV_NAME, 'w') as f:
        f.write('nq,topk,recall\n')
        for i in range(nq):
            line = str(i + 1) + ',' + str(top_k) + ',' + str(recalls[i] * 100) + "%"
            f.write(line + '\n')
        f.write("avarage accuracy:" + str(round(count_all / nq / top_k, 3) * 100) + "%\n")
        f.write("max accuracy:" + str(max(recalls) * 100) + "%\n")
        f.write("min accuracy:" + str(min(recalls) * 100) + "%\n")
    print("total accuracy", round(count_all / nq / top_k, 3) * 100, "%")


def save_pg_re(ids, nq, top_k, table_name):
    filename = CM_FOLDER_NAME + '/' + table_name + CM_GET_LOC_NAME
    conn = connect_postgres_server()
    cur = conn.cursor()
    with open(filename, 'w') as f:
        for i in range(nq):
            for k in range(top_k):
                index = search_loc_in_pg(table_name, cur, ids[i * top_k + k])
                # print(index)
                f.write(index + '\n')
            f.write('\n')


def compare_correct(nq, top_k, rand, loc_gt, loc_se, topk_ground_truth):
    recalls = []
    count_all = 0
    for i in range(nq):
        results = []
        ground_truth = []
        for j in range(top_k):
            results += loc_se[i * top_k + j]
            ground_truth += loc_gt[int(rand[i * top_k]) * topk_ground_truth + j]
        union = list(set(results).intersection(set(ground_truth)))
        count = len(union)
        recalls.append(count/top_k)
        count_all += count
    print("topk_ground_truth:", topk_ground_truth)
    return recalls, count_all


def get_recalls_loc(nq, top_k, rand, table_name):
    loc_gt = load_gt_out(table_name, True)
    loc_se = load_gt_out(table_name, False)
    recalls, count_all = compare_correct(nq, top_k, rand, loc_gt, loc_se, GT_TOPK)
    save_compare_csv(nq, top_k, recalls, count_all, table_name)


def compare_pg_loc(table_name):
    rand, ids, dis, nq = load_search_out(table_name)
    top_k = int(len(rand) / nq)
    print("nq:", nq, "top_k:", top_k)
    time_start = time.time()
    save_pg_re(ids, nq, top_k, table_name)
    time_end = time.time()
    print("time cost in pg search:", round(time_end - time_start, 4))
    get_recalls_loc(nq, top_k, rand, table_name)


def save_re(ids, top_k, table_name):
    filename_id = IDMAP_FOLDER_NAME + '/' + table_name + IDMAP_NAME
    filename = CM_FOLDER_NAME + '/' + table_name + CM_GET_LOC_NAME
    with open(filename, 'w') as f:
        for i in range(len(ids)):
            output = os.popen('./get_id.sh' + ' ' + ids[i] + ' ' + filename_id)
            index = output.read()
            f.write(index)
            if (i+1) % top_k==0:
                f.write('\n')


def compare_loc(table_name):
    rand, ids, dis, nq = load_search_out(table_name)
    top_k = int(len(rand) / nq)
    print("nq:", nq, "top_k:", top_k)
    save_re(ids, top_k, table_name)
    get_recalls_loc(nq, top_k, rand, table_name)



def get_file_loc_txt(cm_file, fnames_file):
    cm1_file = CM_FOLDER_NAME + '/' + cm_file
    se1_file = SE_FOLDER_NAME + '/' + fnames_file
    filenames = os.listdir(BASE_FOLDER_NAME)
    filenames.sort()
    with open(cm1_file, 'r') as cm_f:
        with open(se1_file, 'w') as se_f:
            for line in cm_f:
                if line != '\n':
                    line = line.strip()
                    loc = int(line[0:3])
                    offset = int(line[3:9])
                    se_f.write(filenames[loc] + ' ' + str(offset + 1) + '\n')
                else:
                    se_f.write('\n')

def get_search_file(table_name):
    rand, ids, dis, nq = load_search_out(table_name)
    top_k = int(len(rand) / nq)
    print("nq:", nq, "top_k:", top_k)
    time_start = time.time()
    if PG_LOC | PG_VEC:
        save_pg_re(ids, nq, top_k, table_name)
    else:
        save_re(ids, top_k, table_name)
    get_file_loc_txt(table_name+CM_GET_LOC_NAME, table_name+SE_CM_FILE_NAME)
    time_end = time.time()
    print("time cost :", round(time_end - time_start, 4))


def main():
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "hpft:",
            ["help", "table=", "compare", "file"]
        )
    except getopt.GetoptError:
        print("Usage: python milvus_compare.py --table=<table_name> -p")
        sys.exit(2)

    for opt_name, opt_value in opts:
        if opt_name in ("-h", "--help"):
            print("python milvus_compare.py --table=<table_name> -p")
            sys.exit()
        elif opt_name in ("-t", "--table"):
            table_name = opt_value
        elif opt_name in ("-f", "--file"):  # python3 milvus_compare.py --table=<table_name> -f
            get_search_file(table_name)
        elif opt_name in ("-p", "--compare"):  # python3 milvus_compare.py --table=<table_name> -p
            if not os.path.exists(CM_FOLDER_NAME):
                os.mkdir(CM_FOLDER_NAME)
            if PG_VEC:
                print("compare with postgres's vectors")
                compare_pg_vec(table_name)
            elif PG_LOC:
                print("compare with postgres's location")
                compare_pg_loc(table_name)
            else:
                print("compare with location")
                compare_loc(table_name)


if __name__ == "__main__":
    main()
