import getopt
import sys

from milvus import *

SERVER_ADDR = "192.168.1.10"
SERVER_PORT = 19530

milvus = Milvus()

milvus.connect(SERVER_ADDR, SERVER_PORT)


def main(argv):
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "d:t:ceshpgb",
            ["dim=", "table=", "create", "describe", "show", "has", "drop", "get", "build"],
        )
        # print(opts)
    except getopt.GetoptError:
        print("Usage: test.py -n <nb>  -m <num> -g -t -s")
        sys.exit(2)

    for opt_name, opt_value in opts:
        if opt_name in ("-d", "--dim"):
            dimension = int(opt_value)
        elif opt_name in ("-t", "--table"):
            table_name = opt_value
        elif opt_name in ("-c", "--create"):
            param = {'table_name': table_name, 'dimension': dimension, 'index_type': IndexType.IVF_SQ8}
            print(milvus.create_table(param))
        elif opt_name in ("-e", "--describe"):
            status, table = milvus.describe_table(table_name)
            print(status)
            print(table)
        elif opt_name in ("-s", "--show"):
            status, tables = milvus.show_tables()
            print(status)
            print(tables)
        elif opt_name in ("-h", "--has"):
            print(milvus.has_table(table_name))
        elif opt_name in ("-p", "--drop"):
            print(milvus.delete_table(table_name))
        elif opt_name in ("-g", "--get"):
            result = milvus.get_table_row_count(table_name)
            print(result)
        elif opt_name in ("-b", "--build"):
            print(milvus.build_index(table_name))
        else:
            print("wrong param")
            sys.exit(2)


if __name__ == "__main__":
    main(sys.argv[1:])
