import numpy as np
from multiprocessing import cpu_count, Pool
import pandas as pd
import numpy as np
# cpu 数量
cores = cpu_count()
# 分块个数
partitions = cores


def parallelize(df, func):
    data_split = np.array_split(df, partitions)
    # 进程池
    pool = Pool(cores)
    # 数据合分，并发
    data = pd.concat(pool.map(func, data_split))
    pool.close()

    pool.join()
    return data