#!/usr/bin/env python3
import sys
sys.path.insert(0, "/root/lab/rcd/pyAgrum")
sys.path.insert(0, "/root/lab/rcd/causallearn")

import time
import argparse

import numpy as np

import utils as u

VERBOSE = False

K = 5
SEED = 42
BINS = None

# LOCAL_ALPHA has an effect on execution time. Too strict alpha will produce a sparse graph
# so we might need to run phase-1 multiple times to get up to k elements. Too relaxed alpha
# will give dense graph so the size of the separating set will increase and phase-1 will
# take more time.
# We tried a few different values and found that 0.01 gives the best result in our case
# (between 0.001 and 0.1).
LOCAL_ALPHA = 0.01
DEFAULT_GAMMA = 5

# SRC_DIR = 'sock-shop-data/carts-mem/1/' /home/wangrunzhou/0_warlock/rcd/
SRC_DIR = 'data/s-42/n-10-d-3-an-1-nor-s-1000-an-s-1000/' #'/root/lab/rcd/data/s-42/n-10-d-3-an-1-nor-s-1000-an-s-1000/'

# Split the dataset into multiple subsets，按照列划分的
def create_chunks(df, gamma):
    chunks = list()
    names = np.random.permutation(df.columns) # 按照第一个索引洗牌，产生一个随机序列
    for i in range(df.shape[1] // gamma + 1):
        chunks.append(names[i * gamma:(i * gamma) + gamma])

    # 防御性代码
    if len(chunks[-1]) == 0:
        chunks.pop()
    return chunks

def run_level(normal_df, anomalous_df, gamma, localized, bins, verbose):
    ci_tests = 0
    # print(gamma)
    chunks = create_chunks(normal_df, gamma) #返回一个列名的二维list
    # print(chunks)
    #根据gamma按列切块，这里gamma是5，所以切成五列五列的块儿

    if verbose:
        print(f"Created {len(chunks)} subsets")

    f_child_union = list()
    mi_union = list()
    f_child = list()
    for c in chunks: 
        # Try this segment with multiple values of alpha until we find at least one node
        # 注意，这里执行的已经不是本文件内的top_k_rc了，而是util里的top_k_rc
        # print('这里是run_level啊')

        # rc是f之外的节点按照p从小到大排列的列表，表示目前最可能的根因节点列表，排序由强到弱
        # mi是删除的边，来自PC的skeleton构建过程
        rc, _, mi, ci = u.top_k_rc(normal_df.loc[:, c],
                                   anomalous_df.loc[:, c],
                                   bins=bins,
                                   localized=localized,
                                   start_alpha=LOCAL_ALPHA,
                                   min_nodes=1,
                                   verbose=verbose)
        f_child_union += rc
        mi_union += mi
        ci_tests += ci
        if verbose:
            f_child.append(rc)

    if verbose:
        print(f"Output of individual chunk {f_child}")
        print(f"Total nodes in mi => {len(mi_union)} | {mi_union}")

    return f_child_union, mi_union, ci_tests

# 怎么个多阶段？
def run_multi_phase(normal_df, anomalous_df, gamma, localized, bins, verbose):
    f_child_union = normal_df.columns
    mi_union = []
    i = 0
    prev = len(f_child_union)

    # Phase-1
    ci_tests = 0
    while True:
        start = time.time()
        f_child_union, mi, ci = run_level(normal_df.loc[:, f_child_union], #所有行保留，筛选列
                                          anomalous_df.loc[:, f_child_union],
                                          gamma, localized, bins, verbose)
        if verbose:
            print(f"Level-{i}: variables {len(f_child_union)} | time {time.time() - start}")
        i += 1
        mi_union += mi
        ci_tests += ci
        # Phase-1 with only one level
        # break

        len_child = len(f_child_union)
        # If found gamma nodes or if running the current level did not remove any node
        # 每一轮只对上一轮中f的子节点进行run_level，当：
        # 1.剩余的节点数目小于gamma，或；
        # 2.特殊情况下len_child == prev（说明独立检验再也没法去除节点了）；
        # 这两种情况下返回；
        if len_child <= gamma or len_child == prev: break
        prev = len(f_child_union)

    # print(f_child_union)
    # Phase-2
    # 这时，多数情况下返回的节点数小于等于gamma，最后执行一次top_k_rc即可
    # gamma似乎也是
    mi_union = []
    new_nodes = f_child_union
    rc, _, mi, ci = u.top_k_rc(normal_df.loc[:, new_nodes],
                               anomalous_df.loc[:, new_nodes],
                               bins=bins,
                               mi=mi_union,
                               localized=localized,
                               verbose=verbose)
    ci_tests += ci
    return rc, ci_tests

def rca_with_rcd(normal_df, anomalous_df, bins,
                 gamma=DEFAULT_GAMMA, localized=False, verbose=VERBOSE):
    start = time.time()
    rc, ci_tests = run_multi_phase(normal_df, anomalous_df, gamma, localized, bins, verbose)
    end = time.time()

    return {'time': end - start, 'root_cause': rc, 'tests': ci_tests}

def top_k_rc(normal_df, anomalous_df, k, bins,
             gamma=DEFAULT_GAMMA, seed=SEED, localized=False, verbose=VERBOSE):
    np.random.seed(seed)
    result = rca_with_rcd(normal_df, anomalous_df, bins, gamma, localized, verbose)
    return {**result, 'root_cause': result['root_cause'][:k]}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run PC on the given dataset')

    parser.add_argument('--path', type=str, default=SRC_DIR,
                        help='Path to the experiment data')
    parser.add_argument('--k', type=int, default=K,
                        help='Top-k root causes')
    parser.add_argument('--local', action='store_true',
                        help='Run localized version to only learn the neighborhood of F-node')

    args = parser.parse_args()
    path = args.path
    k = args.k
    local = args.local

    #俩df
    (normal_df, anomalous_df) = u.load_datasets(path + 'normal.csv',
                                                path + 'anomalous.csv')

    # Enable the following line for sock-shop or real outage dataset
    # normal_df, anomalous_df = u.preprocess(normal_df, anomalous_df, 90)

    result = top_k_rc(normal_df, anomalous_df, k=k, bins=BINS, localized=local)
    print(f"Top {k} took {round(result['time'], 4)} and potential root causes are {result['root_cause']} with {result['tests']} tests")
