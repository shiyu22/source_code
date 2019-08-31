import sys, getopt
from collections import defaultdict
import pandas as pd
import numpy  as np
import time
import random
import os
import glob
from multiprocessing import Process
from milvus import Milvus, Prepare, IndexType, Status

MILVUS = Milvus()
SERVER_ADDR = "127.0.0.1"
SERVER_PORT = 19530
NQ = 0


# IP = False
# L2 = True
IP = True
L2  =False

CSV = False
# CSV = True
UINT8 = False
# UINT8 = True

# have to set the query's folder name
NQ_FOLDER_NAME = '/data/shiyu/data'
BASE_FOLDER_NAME = '/data/shiyu/data'
GT_FOLDER_NAME = 'gt_all'
RE_FOLDER_NAME = 'gt_output'
GT_NAME = '_ground_truth.txt'
GT_FNAMES = '_gt_file.txt'

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


# get vectors of the files
def load_nq_vec(nq):
    vectors = []
    filenames = os.listdir(NQ_FOLDER_NAME)  # get the whole file names
    filenames.sort()
    if nq == 0:
        for filename in filenames:
            vectors += load_vec_list(NQ_FOLDER_NAME + '/' +filename)
        return vectors

    length = 0
    flag = 0
    for filename in filenames:
        vec_list = load_vec_list(NQ_FOLDER_NAME + '/' +filename)
        length += len(vec_list)
        if(length>nq):
            flag =1
            num = length%nq
            if (num!=0 and flag==0):
                vec_list = load_vec_list(NQ_FOLDER_NAME + '/' +filename,num)
            else:
                vec_list = load_vec_list(NQ_FOLDER_NAME + '/' +filename,nq)
        vectors += vec_list
        if len(vectors)==nq :
            return vectors

# load vectors from filr_name and num means nq's number
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


# Calculate the European distance
def calEuclideanDistance(vec1,vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        dist = np.sqrt(np.sum(np.square(vec1 - vec2)))
        return dist

def calInnerDistance(vec1,vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        dist = np.inner(vec1,vec2)
        return dist

def get_ground_truth_l2(table_name, nq, topk ,idx,vct_nq):
    filenames = os.listdir(BASE_FOLDER_NAME)  # get the whole file names
    filenames.sort()
    no_dist = {}
    re = []
    k = 0 
    for filename in filenames:
        vec_list = load_vec_list(BASE_FOLDER_NAME+'/'+filename)
        for j in range(len(vec_list)):
            dist = calEuclideanDistance(vct_nq,vec_list[j])
            num_j = "%03d%06d" % (k,j)
            if j<topk and k==0:
                no_dist[num_j] =  dist
            else:
                #sorted by values
                max_key = max(no_dist,key=no_dist.get)
                
                max_value = no_dist[max_key]
                if(dist < max_value):
                    m = no_dist.pop(max_key)
                    no_dist[num_j] =  dist
        k = k+1
    no_dist = sorted(no_dist.items(), key=lambda x: x[1])
    # print("topk:",topk)
    # print("len_no_dist:",len(no_dist))
    save_gt_file(table_name,no_dist,idx)

def get_ground_truth_ip(table_name, nq, topk ,idx,vct_nq):
    filenames = os.listdir(BASE_FOLDER_NAME)  # get the whole file names
    filenames.sort()
    no_dist = {}
    re = []
    k = 0 
    for filename in filenames:
        vec_list = load_vec_list(BASE_FOLDER_NAME+'/'+filename)
        for j in range(len(vec_list)):
            dist = calInnerDistance(vct_nq,vec_list[j])
            num_j = "%03d%06d" % (k,j)
            if j<topk and k==0:
                no_dist[num_j] =  dist
            else:
                #sorted by values
                min_key = min(no_dist,key=no_dist.get)
                
                min_value = no_dist[min_key]
                if(dist > min_value):
                    m = no_dist.pop(min_key)
                    no_dist[num_j] =  dist
        k = k+1
    no_dist = sorted(no_dist.items(), key=lambda x: x[1], reverse=True)
    # print("topk:",topk)
    # print("len_no_dist:",len(no_dist))
    save_gt_file(table_name,no_dist,idx)

def save_gt_file(table_name, no_dist, idx):
    s = "%05d" % idx
    idx_fname = './' + table_name + '_' + GT_FOLDER_NAME + '/' + s + '_idx.txt'
    dis_fname = './' + table_name + '_' + GT_FOLDER_NAME + '/' + s + '_dis.txt'
    with open(idx_fname,'w') as f:
        for re in no_dist:
            f.write(str(re[0])+'\n')
        f.write('\n')
    with open(dis_fname,'w') as f:
        for re in no_dist:
            f.write(str(re[1])+'\n')
        f.write('\n')

def get_ground_truth_txt(table_name, file):
    filenames = os.listdir('./' + table_name + '_' + GT_FOLDER_NAME)
    filenames.sort()
    write_file = open(RE_FOLDER_NAME + '/' + table_name + file,'w+')
    for f in filenames:
        if f.endswith('_idx.txt'):
            f = './' + table_name + '_' + GT_FOLDER_NAME + '/' + f
            for line in open(f,'r'):
                write_file.write(line)

def get_ground_truth_fname_txt(table_name, gt_file, fnames_file):
    gt_file = RE_FOLDER_NAME + '/' + table_name + gt_file
    fnames_file = RE_FOLDER_NAME + '/' + table_name + fnames_file
    filenames = os.listdir(BASE_FOLDER_NAME)
    filenames.sort()
    with open(gt_file,'r') as gt_f:
        with open(fnames_file,'w') as fnames_f:
            for line in gt_f:
                if line != '\n':
                    line = line.strip()
                    loca = int(line[0:3])
                    offset = int(line[3:9])
                    fnames_f.write(filenames[loca]+' '+str(offset+1) + '\n')
                else:
                    fnames_f.write('\n')

def ground_truth_process(table_name, nq, topk):
    try:
        os.mkdir('./' + table_name + '_' + GT_FOLDER_NAME)
    except:
        print('There already exits folder named ' + table_name + '_' + GT_FOLDER_NAME + '!')
    else:
        vectors = load_nq_vec(nq)
        print("query list:",len(vectors))
        processes = []
        process_num = 8
        nq = len(vectors)
        loops = nq // process_num
        rest = nq % process_num
        if rest != 0:
            loops += 1
        time_start = time.time()
        for loop in range(loops):
            time1_start = time.time()
            base = loop * process_num
            if rest!=0 and loop == loops-1:
                process_num = rest
            print('base:',loop)
            for i in range(process_num):
                print('nq_index:', base+i)
                # seed = np.random.RandomState(base+i)
                if L2:
                    if base+i==0:
                        print("get groun_truth by L2.")
                    process = Process(target=get_ground_truth_l2, args=(table_name, nq, topk, base+i ,vectors[base+i]))
                elif IP:
                    if base+i==0:
                        print("get groun_truth by IP.")
                    process = Process(target=get_ground_truth_ip, args=(table_name, nq, topk, base+i ,vectors[base+i]))
                processes.append(process)
                process.start()
            for p in processes:
                    p.join()
            time1_end = time.time()
            print("base", loop, "time_cost = ",round(time1_end - time1_start,4))
        if not os.path.exists(RE_FOLDER_NAME):
            os.mkdir(RE_FOLDER_NAME)
        get_ground_truth_txt(table_name, GT_NAME)
        get_ground_truth_fname_txt(table_name, GT_NAME, GT_FNAMES)
        time_end = time.time()
        time_cost = time_end - time_start
        print("total_time = ",round(time_cost,4),"\nGet the ground truth successfully!")

def main():
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "hlq:k:t:",
            ["help","table=","nq=","topk="],
        )
    except getopt.GetoptError:
        print("Usage: test.py --table <table_name> [-q <nq>] -k <topk> -s")
        sys.exit(2)
    nq = 0
    for opt_name, opt_value in opts:
        if opt_name in ("-h", "--help"):
            print("test.py -table <table_name> [-q <nq>] -k <topk> -l")
            sys.exit()
        elif opt_name in ("-t", "--table"):
            table_name = opt_value
        elif opt_name in ("-q", "--nq"):
            nq = int(opt_value)
        elif opt_name in ("-k", "--topk"):
            topk = int(opt_value)
        elif opt_name == "-l":
            connect_server()
            ground_truth_process(table_name,nq,topk)
            sys.exit()
if __name__ == '__main__':
    main()
