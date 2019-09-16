# milvus_ground_truth.py说明:

### milvus_ground_truth.py参数说明：

| 参数               | 描述                             | 默认设置             |
| ------------------ | -------------------------------- | -------------------- |
| GET_VEC            | 是否保存为npy格式的向量          | False                |
| PROCESS_NUM        | 脚本执行的进程数                 | 5                    |
| IP                 | Milvus的metric_type是否为IP      | True                 |
| L2                 | Milvus的metric_type是否为L2      | False                |
| CSV                | 查询向量文件格式是否为.csv       | False                |
| UINT8              | 查询向量是否为uint8格式          | False                |
| BASE_FOLDER_NAME   | 源向量数据集的路径               | '/data/milvus'       |
| NQ_FOLDER_NAME     | 查询向量集的路径                 | '/data/milvus/query' |
| GT_ALL_FOLDER_NAME | 执行多进程产生的中间文件         | 'ground_truth_all'   |
| GT_FOLDER_NAME     | gorund truth结果保存的路径       | 'ground_truth'       |
| LOC_FILE_NAME      | 该文件存储gorund truth的位置信息 | 'location.txt'       |
| FLOC_FILE_NAME     | 该文件存储gorund truth的文件信息 | 'file_location.txt'  |
| VEC_FILE_NAME      | 该文件存储gorund truth向量       | 'vectors.npy'        |

### milvus_ground_truth.py使用说明：

```bash
python3 milvus_ground_truth.py --table <table_name> [-q <nq_num>] -k <topk_num> -l

# 执行-l生成源向量数据集的ground truth，并将结果写入GT_FOLDER_NAME目录,其中LOC_FILE_NAME存储着数字位置，如"002005210"，FLOC_FILE_NAME中存着结果对应的文件名和位置，如"binary_128d_00000.npy 81759"
# -q或者--nq表示从查询向量集中按序取出的向量个数，该参数可选，若没有-q表示ground truth需查询向量为查询集中的全部数据
# -k或者--topk表示ground truth需查询每个向量的前k个相似的向量
```



# milvus_search.py说明:

### milvus_search.py参数说明：

| 参数           | 描述                       | 默认设置       |
| -------------- | -------------------------- | -------------- |
| SERVER_ADDR    | Milvus的IP设置             | "127.0.0.1"    |
| SERVER_PORT    | Milvus的端口设置           | 19530          |
| NQ_FOLDER_NAME | 查询向量集的路径           | '/data/milvus' |
| SE_FOLDER_NAME | 查询结果保存的路径         | 'search'       |
| SE_FILE_NAME   | 查询结果保存的文件         | '_output.txt'  |
| **GT_NQ**      | **ground truth中的nq数值** | **0**          |
| CSV            | 查询向量文件格式是否为.csv | False          |
| UINT8          | 查询向量是否为uint8格式    | False          |

### milvus_search.py使用说明：

```bash
python3 milvus_search.py --table <table_name> [-q <nq_num>] -k <topk_num> -s

# 执行-s实现Milvus的向量查询，并将结果写入SEARCH_FOLDER_NAME目录下的table_name_output.txt中，该文件有随机数，查询结果ids和查询结果distance三列
# -t或者--table表示需要查询的表名
# -q或者--nq表示在查询集中随机选取的查询向量个数，该参数可选，若没有-q表示查询向量为查询集中的全部数据
# -k或者--topk表示查询每个向量的前k个相似的向量
```





# milvus_compare.py说明：

### milvus_compare.py参数说明：
| 参数 | 描述 | 默认设置 |
| ----------------- | ---- | -------------------- |
| PG_HOST           | postgres的IP设置 | ’localhost‘          |
| PG_PORT           | postgres的端口设置 | 5432                 |
| PG_USER           | postgres的用户设置 | ’postgres‘           |
| PG_PASSWORD       | postgres的用户密码 | ‘postgres’           |
| PG_DATABASE       | postgres的数据库名称 | ’postgres‘           |
| PG_VEC            | 是否用postrgres向量进行准确率比较 | TRUE                 |
| PG_LOC            | 是否用postrgres位置进行准确率比较 | TRUE                 |
| UINT8             | ground_truth向量是否为uint8格式 | TRUE                 |
| GT_TOPK           | ground_truth中topk数值 | 22                   |
| BASE_FOLDER_NAME  | 源向量数据集的路径 | 'E:/BaiduPan/data_3' |
| GT_FOLDER_NAME    | gorund truth结果保存的路径 | 'ground_truth'       |
| SE_FOLDER_NAME    | 查询结果保存的路径 | 'search'             |
| SE_CM_FILE_NAME   | 该文件存储查询结果的文件位置 | '_file_output.txt'   |
| CM_FOLDER_NAME    | 准备率比较结果保存的路径 | 'compare'            |
| IDMAP_FOLDER_NAME | 加载数据结果保存的路径 | 'idmap'              |
| IDMAP_NAME        | 该文件存储加载数据结果的ids和位置 | '_idmap.txt'         |
| GT_NAME           | 该文件存储gorund truth的位置信息 | 'location.txt'       |
| GT_FILE_NAME      | 该文件存储gorund truth的文件信息 | 'file_location.txt'  |
| GT_VEC_NAME       | 该文件存储gorund truth向量 | 'vectors.npy'        |
| SE_FILE_NAME      | 查询结果保存的文件 | '_output.txt'        |
| CM_CSV_NAME       | 该文件存储准确率比较结果 | '_output.csv'        |
| CM_GET_LOC_NAME   | 该文件存储查询结果的位置信息 | '_loc_compare.txt'   |

### milvus_compare.py使用说明：

```bash
python3 milvus_compare.py --table=<table_name> -p

# 执行-p或者--compare实现准确率比较，在CM_FOLDER_NAME目录下生成结果table_nq_topk_CM_CSV_NAME, table表示表名, nq/topk表示查询时的nq/topk, 该文件由三列组成，recall表示查询结果与ground truth相比较的准确率，最后显示平均准确率，准确率的最大值和最小值。
# -t或者--table表示需要查询的表名

python3 milvus_compare.py --table=<table_name> -f

# 执行-f或者--file对milvus查询结果处理，返回查询结果对应的向量文件位置，将在SE_FOLDER_NAME目录下生成SE_CM_FILE_NAME，其存储查询结果对应的文件名和向量位置信息，如"binary_128d_00000.npy 1"。
# -t或者--table表示需要查询的表名
```


# milvus_toolkit.py说明:

## 1、参数说明

| 参数           | 描述                      | 默认设置                    |
| -------------- | ------------------------- | --------------------------- |
| SERVER_ADDR    | Milvus的IP设置            | "0.0.0.0"                   |
| SERVER_PORT    | Milvus的端口设置          | 19530                       |
| NL_FOLDER_NAME | 归一化向量数据的存放路径  | /data/workspace/data/data_2 |
| NQ_FOLDER_NAME | 查询向量集的存放路径      | /data/workspace/data/data_2 |
| IS_CSV         | 数据文件是否以csv格式存储 | False                       |
| IS_UINT8       | 数据格式是否为uint8       | True                        |

## 2、使用说明

| 功能       | 说明                 | 举例                                                         |
| ---------- | -------------------- | ------------------------------------------------------------ |
| -c         | 在milvus中建表       | python3 milvus_toolkit.py --table=test <br />--dim=512 --index=sq8  -c |
| --show     | 显示milvus中的表     | python3  milvus --show                                       |
| --describe | 显示某张表的具体信息 | python3 milvus_toolkit.py --table=test <br />--describe      |
| -d         | 删除milvus中的某张表 | python3 milvus_toolkit.py --table=test --d                   |
| --rows     | 显示某张表的的行数   | python3 milvus_tookit.py --table=test                        |
| --build    | 为某张表建立索引     | python3 milvus_toolkit.py --table=test --build               |
| -n         | 判断数据是否为归一化 |                                                              |
|            |                      |                                                              |


