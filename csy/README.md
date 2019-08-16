# milvus_search.py说明:

### milvus_search.py参数说明：

| 参数           | 描述                       | 默认设置       |
| -------------- | -------------------------- | -------------- |
| SERVER_ADDR    | Milvus的IP设置             | "127.0.0.1"    |
| SERVER_PORT    | Milvus的端口设置           | 19530          |
| NQ_FOLDER_NAME | 查询向量集的绝对路径       | '/data/milvus' |
| NQ             | 查询向量集中的向量个数     | 0              |
| CSV            | 查询向量文件格式是否为.csv | False          |
| UINT8          | 查询向量是否为uint8格式    | False          |

### milvus_search.py使用说明：

```bash
python3 milvus_search.py --table <table_name> [-q <nq_num>] -k <topk_num> -s

# 执行-s实现Milvus的向量查询，并将结果写入search_optput目录下的table_name_output.txt中，该文件有随机数，查询结果ids和查询结果distance三列
# -t或者--table表示需要查询的表名
# -q或者--nq表示在查询集中随机选取的查询向量个数，该参数可选，若没有-q表示查询向量为查询集中的全部数据
# -k或者--topk表示查询每个向量的前k个相似的向量
```



# milvus_ground_truth.py说明:

### milvus_ground_truth.py参数说明：

| 参数             | 描述                        | 默认设置             |
| ---------------- | --------------------------- | -------------------- |
| SERVER_ADDR      | Milvus的IP设置              | "127.0.0.1"          |
| SERVER_PORT      | Milvus的端口设置            | 19530                |
| IP               | Milvus的metric_type是否为IP | True                 |
| L2               | Milvus的metric_type是否为L2 | False                |
| BASE_FOLDER_NAME | 源向量数据集的绝对路径      | '/data/milvus'       |
| NQ_FOLDER_NAME   | 查询向量集的绝对路径        | '/data/milvus/query' |
| CSV              | 查询向量文件格式是否为.csv  | False                |
| UINT8            | 查询向量是否为uint8格式     | False                |

### milvus_ground_truth.py使用说明：

```bash
python3 milvus_ground_truth.py --table <table_name> [-q <nq_num>] -k <topk_num> -l

# 执行-l生成源向量数据集的ground truth，并将结果写入gt_optput目录,其中table_name_ground_truth.txt存储着数字位置，如"002005210"，table_name_gt_file.txt中存着结果对应的文件名和位置，如"binary_128d_00000.npy 81759"
# -t或者--table表示源向量数据集对应的表名
# -q或者--nq表示从查询向量集中按序取出的向量个数，该参数可选，若没有-q表示ground truth需查询向量为查询集中的全部数据
# -k或者--topk表示ground truth需查询每个向量的前k个相似的向量
```

