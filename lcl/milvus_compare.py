import getopt
import sys
import psycopg2

PG_HOST = "192.168.1.101"
PG_PORT = 5432
PG_USER = "zilliz"
PG_PASSWORD = "Fantast1c"
PG_DATABASE = "postgres"
PG_TABLE = 'deep_1b_sq8_idmap'

FANME = '/mnt/data/deep1B_groundtruth.ivecs'

def load_search_out():
    fname = '/root/search_output/deep_1B_sq8_output.txt'
    ids = []
    rand = []
    distance = []
    with open(fname,'r') as f:
        for line in f.readlines():
            data = line.split()
            if data:
                rand.append(data[0])
                ids.append(data[1])
                distance.append(data[2])
    return ids, rand, distance

def get_location_frpg(list):



def get_location_frgt(list):



def caculate_accu(l1,l2):


def compare_results(table_name):
    search_ids = load_search_out()[1]
    rand_ids = load_search_out()[0]
    search_location = get_location_frpg(search_ids)
    gt_location = get_location_frgt(rand_ids)
    accu = caculate_accu(search_location,gt_location)
    return  accu



def main():
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "hp",
            ["help", "table=", "compare"]
        )
    except getopt.GetoptError:
        print( "Usage: python milvus_compare.py --table=<table_name> -p" )
        sys.exit( 2 )

    for opt_name, opt_value in opts:
        if opt_name in ("-h", "--help"):
            print( "python milvus_compare.py --table=<table_name> -p" )
            sys.exit()
        elif opt_name == "--table":
            table_name = opt_value
        elif opt_name in ("-p", "--compare"):
            print("search accuracy:" + compare_results(table_name) )


if __name__ == "__main__":
    main()
