import psycopg2
import numpy as np

PG_HOST = "192.168.1.101"
PG_PORT = 5432
PG_USER = "zilliz"
PG_PASSWORD = "Fantast1c"
PG_DATABASE = "postgres"
PG_TABLE = 'deep_1b_sq8_idmap'

FANME = '/mnt/data/deep1B_groundtruth.ivecs'


def load_gt_result():
    x = np.memmap(FANME, dtype='int32', mode='r')
    d = x[0]
    data = x.reshape(-1, d + 1)[:, 1:]
    data = data.tolist()
    return data


def load_search_out():
    fname = '/root/search_output/deep_1B_sq8_output.txt'
    ids = []
    rand = []
    distance = []
    with open(fname, 'r') as f:
        for line in f.readlines():
            data = line.split()
            if data:
                rand.append(data[0])
                ids.append(data[1])
                distance.append(data[2])
    return ids, rand, distance


def connect_postgres_server():
    try:
        conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, database=PG_DATABASE)
        print("connect the database!")
        return conn
    except:
        print("unable to connect to the database")


def search_in_pg(conn, cur, id):
    sql = "select location from " + PG_TABLE + " where ids = " + id + ";"
    # print(sql)
    # print(id)
    try:
        cur.execute(sql)
        rows = cur.fetchall()
        return rows
    except:
        print("search faild!")


def compare(ids, gt_result, rand):
    correct = []
    count = 0
    conn = connect_postgres_server()
    cur = conn.cursor()
    search_result = []
    for i in range(len(ids)):
        rows = search_in_pg(conn, cur, ids[i])
        search_result.append(rows[0][0])
        # print(rows[0][0])
        # print(gt_result[int(rand[i])][0])
        if rows[0][0] == gt_result[int(rand[i])][0]:
            correct.append("100%")
            count = count + 1
        else:
            correct.append("0%")
    correct_all = count / (len(ids))
    return correct_all, correct, search_result


def record_result(correct, correct_all, gt_result, search_result, rand, distance):
    fname = str(len(rand)) + '_result_all.csv'
    filename = '/root/output_result/' + fname
    with open(filename, 'w') as f:
        f.write('topk,远程ID,基准ID,搜索结果,distance,recall' + '\n')
        for i in range(len(rand)):
            line = "1," + rand[i] + "," + str(gt_result[int(rand[i])][0]) + "," + str(search_result[i]) + "," + str(
                distance[i]) + "," + str(correct[i]) + "\n"
            f.write(line)
        f.write('total accuracy: ' + str(correct_all * 100) + '%')
    f.close


def main():
    gt_result = load_gt_result()
    ids, rand, distance = load_search_out()
    correct_all, correct, search_result = compare(ids, gt_result, rand)
    record_result(correct, correct_all, gt_result, search_result, rand, distance)
    print("compare end!")
    # record_result(ids)


if __name__ == "__main__":
    main()
