from itertools import combinations

import numpy as np
from tqdm.auto import tqdm

from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.cit import chisq, gsq
from causallearn.utils.PCUtils.Helper import append_value


def skeleton_discovery(data, alpha, indep_test, stable=True, background_knowledge=None,
                       labels={}, verbose=False, show_progress=True):
    '''
    Perform skeleton discovery

    Parameters
    ----------
    data : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of
            samples and n_features is the number of features.
    alpha: float, desired significance level of independence tests (p_value) in (0,1)
    indep_test : the function of the independence test being used
            [fisherz, chisq, gsq, mv_fisherz, kci]
           - fisherz: Fisher's Z conditional independence test
           - chisq: Chi-squared conditional independence test
           - gsq: G-squared conditional independence test
           - mv_fisherz: Missing-value Fishers'Z conditional independence test
           - kci: Kernel-based conditional independence test
    stable : run stabilized skeleton discovery if True (default = True)
    background_knowledge : background knowledge
    verbose : True iff verbose output should be printed.
    show_progress : True iff the algorithm progress should be show in console.

    Returns
    -------
    cg : a CausalGraph object. Where cg.G.graph[j,i]=0 and cg.G.graph[i,j]=1 indicates  i -> j ,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i -- j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.

    '''

    assert type(data) == np.ndarray
    assert 0 < alpha < 1

    no_of_var = data.shape[1]
    # print(no_of_var) #6,五个节点加一个f节点

    cg = CausalGraph(no_of_var, labels=labels)
    # print(cg.G.graph) # node_map长度为6，合格 #内部都是-1，对角线是0，说明没有自环

    cg.set_ind_test(indep_test) #这里indep_test应该是卡方检验，这个函数将图的假设检验方法固定为卡方检验

    cg.data_hash_key = hash(str(data)) #确定data的哈希键值

    # print(data)

    if indep_test == chisq or indep_test == gsq:
        # if dealing with discrete data, data is numpy.ndarray with n rows m columns,
        # for each column, translate the discrete values to int indexs starting from 0,
        #   e.g. [45, 45, 6, 7, 6, 7] -> [2, 2, 0, 1, 0, 1]
        #        ['apple', 'apple', 'pear', 'peach', 'pear'] -> [0, 0, 2, 1, 2]
        # in old code, its presumed that discrete `data` is already indexed,
        # but here we make sure it's in indexed form, so allow more user input e.g. 'apple' ..
        def _unique(column):
            return np.unique(column, return_inverse=True)[1]

        cg.is_discrete = True
        cg.data = np.apply_along_axis(_unique, 0, data).astype(np.int64)
        cg.cardinalities = np.max(cg.data, axis=0) + 1
    else:
        cg.data = data

    # print('888888888888888888888888888888888')
    # print(cg.data)
    #这里干了个啥呢？其实就是把f_node列里的字串类型的0和1变成离散的numpy类型的0和1
    #我觉得完全可以在之前的处理中做这一步，但这里多了防御性编码的成分（确保离散），所以不修改了

    # print(cg.max_degree()) #这里的最大度理应是5，没错确实是5

    depth = -1
    pbar = tqdm(total=no_of_var) if show_progress else None #计算进度？

    # no_of_var是6，也即包括f_node的节点数
    while cg.max_degree() - 1 > depth:
        depth += 1  #从0开始？何意味？
        edge_removal = []
        if show_progress: pbar.reset()
        for x in range(no_of_var):
            if show_progress: pbar.update()
            if show_progress: pbar.set_description(f'Depth={depth}, working on node {x}')
            Neigh_x = cg.neighbors(x)
            # print(x,end=' neighbors are: ')
            # print(Neigh_x)

            if len(Neigh_x) < depth - 1: #这里的depth从0开始,但是0没有跳，多了反而会跳
                continue
            
            # 从邻居中选一个y出来
            for y in Neigh_x:

                #这一部分是基于先验知识来ban一些边，目前方法不涉及
                knowledge_ban_edge = False
                sepsets = set()
                if background_knowledge is not None and (
                        background_knowledge.is_forbidden(cg.G.nodes[x], cg.G.nodes[y])
                        and background_knowledge.is_forbidden(cg.G.nodes[y], cg.G.nodes[x])):
                    knowledge_ban_edge = True
                
                if knowledge_ban_edge:
                    if not stable:
                        edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                        if edge1 is not None:
                            cg.G.remove_edge(edge1)
                        edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                        if edge2 is not None:
                            cg.G.remove_edge(edge2)
                        append_value(cg.sepset, x, y, ())
                        append_value(cg.sepset, y, x, ())
                        break
                    else:
                        edge_removal.append((x, y))  # after all conditioning sets at
                        edge_removal.append((y, x))  # depth l have been considered

                #Neigh_x_noy是x的邻居除去了y
                Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))
                # print('x:',x,' y:',y,' Neigh_x_noy:',Neigh_x_noy)
                # print(depth)
                # print('Neigh_x_noy:',Neigh_x_noy, ' comb:',list(combinations(Neigh_x_noy, depth)))

                #这里包括0是因为在没有任何条件的情况下两个节点依然可以独立
                for S in combinations(Neigh_x_noy, depth): #组合的穷举，但是包括0
                    p = cg.ci_test(x, y, S)
                    if p > alpha:
                        if verbose: print('%d ind %d | %s with p-value %f\n' % (x, y, S, p))
                        if not stable:
                            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                            if edge1 is not None:
                                cg.G.remove_edge(edge1)
                            edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                            if edge2 is not None:
                                cg.G.remove_edge(edge2)
                            append_value(cg.sepset, x, y, S)
                            append_value(cg.sepset, y, x, S)
                            break
                        else:
                            edge_removal.append((x, y))  # after all conditioning sets at
                            edge_removal.append((y, x))  # depth l have been considered
                            for s in S:
                                sepsets.add(s)
                    else:
                        append_value(cg.p_values, x, y, p)
                        if verbose: print('%d dep %d | %s with p-value %f\n' % (x, y, S, p))
                append_value(cg.sepset, x, y, tuple(sepsets))
                append_value(cg.sepset, y, x, tuple(sepsets))

        if show_progress: pbar.refresh()

        for (x, y) in list(set(edge_removal)):
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                cg.G.remove_edge(edge1)

    if show_progress: pbar.close()
    # print(cg.G.graph)
    return cg

# 1\基于对称，12检验了就无需检验21
# 2\不要半途删除，到最后再删，确保边的数目随着alpha递增而递增
# local_node就是f_node
# def local_skeleton_discovery(data, local_node, alpha, indep_test, mi=[], labels={}, verbose=False):
#     # print('mi: ',mi) #从头到尾都是空的，写了个寂寞

#     # print('run local skeleton discovery')
#     assert type(data) == np.ndarray
#     assert local_node <= data.shape[1]
#     assert 0 < alpha < 1

#     no_of_var = data.shape[1]
#     cg = CausalGraph(no_of_var, labels=labels)
#     cg.set_ind_test(indep_test)
#     cg.data_hash_key = hash(str(data))
#     if indep_test == chisq or indep_test == gsq:
#         def _unique(column):
#             return np.unique(column, return_inverse=True)[1]

#         cg.is_discrete = True
#         cg.data = np.apply_along_axis(_unique, 0, data).astype(np.int64)
#         cg.cardinalities = np.max(cg.data, axis=0) + 1
#     else:
#         cg.data = data

#     depth = -1
#     x = local_node
#     # Remove edges between nodes in MI and F-node
#     for i in mi:
#         cg.remove_edge(x, i)

#     while cg.max_degree() - 1 > depth:
#         depth += 1
        
#         # 只关注和FNODE（x）相邻的节点，但一开始是完全图，所以全部的FNODE之外的节点都被考虑
#         local_neigh = np.random.permutation(cg.neighbors(x))
#         # local_neigh = cg.neighbors(x)
#         for y in local_neigh:
#             Neigh_y = cg.neighbors(y) #y的邻居
#             Neigh_y = np.delete(Neigh_y, np.where(Neigh_y == x)) #y的邻居删除x（FNODE）
#             Neigh_y_f = [] # 同时和xy相连的节点
#             if depth > 0:
#                 Neigh_y_f = [s for s in Neigh_y if x in cg.neighbors(s)]
#                 # Neigh_y_f += mi

#             for S in combinations(Neigh_y_f, depth):
#                 p = cg.ci_test(x, y, S)
#                 if p > alpha:
#                     if verbose: print(f'{cg.labels[x]} ind {cg.labels[y]} | {[cg.labels[s] for s in S]} with p-value {p}')
#                     cg.remove_edge(x, y)
#                     append_value(cg.sepset, x, y, S)
#                     append_value(cg.sepset, y, x, S)

#                     if depth == 0:
#                         cg.append_to_mi(y)
#                     break
#                 else:
#                     append_value(cg.p_values, x, y, p)
#                     if verbose: print(f'{cg.labels[x]} dep {cg.labels[y]} | {[cg.labels[s] for s in S]} with p-value {p}')

#     return cg



def local_skeleton_discovery(data, local_node, alpha, indep_test, mi=[], labels={}, verbose=False):
    assert isinstance(data, np.ndarray)
    assert local_node < data.shape[1]
    assert 0 < alpha < 1

    no_of_var = data.shape[1]
    cg = CausalGraph(no_of_var, labels=labels)
    cg.set_ind_test(indep_test)
    cg.data_hash_key = hash(str(data))

    # 离散数据预处理
    if indep_test in [chisq, gsq]:
        def _unique(column):
            return np.unique(column, return_inverse=True)[1]
        cg.is_discrete = True
        cg.data = np.apply_along_axis(_unique, 0, data).astype(np.int64)
        cg.cardinalities = np.max(cg.data, axis=0) + 1
    else:
        cg.data = data

    depth = -1
    x = local_node

    # Step 1：记录已独立的边（延后删除）
    indep_edges = set()     # {(i,j), (j,i)} 形式

    # Step 2：先移除 MI（与F节点先验独立）
    for i in mi:
        indep_edges.add(tuple(sorted((x, i))))

    # Step 3：主循环
    while cg.max_degree() - 1 > depth:
        depth += 1

        # 随机打乱顺序（局部探索）
        local_neigh = np.random.permutation(cg.neighbors(x))

        for y in local_neigh:
            if tuple(sorted((x, y))) in indep_edges:
                continue  # 已知独立，不再检验

            Neigh_y = cg.neighbors(y)
            Neigh_y = np.delete(Neigh_y, np.where(Neigh_y == x))

            Neigh_y_f = []
            if depth > 0:
                Neigh_y_f = [s for s in Neigh_y if x in cg.neighbors(s)]

            # 枚举所有条件集合
            for S in combinations(Neigh_y_f, depth):
                p = cg.ci_test(x, y, S)
                append_value(cg.p_values, x, y, p)

                if p > alpha:  # 独立
                    indep_edges.add(tuple(sorted((x, y))))
                    append_value(cg.sepset, x, y, S)
                    append_value(cg.sepset, y, x, S)

                    if depth == 0:
                        cg.append_to_mi(y)
                    if verbose:
                        print(f'{cg.labels[x]} ⟂ {cg.labels[y]} | {[cg.labels[s] for s in S]} (p={p:.4f})')
                    break
                else:
                    if verbose:
                        print(f'{cg.labels[x]} not indep {cg.labels[y]} | {[cg.labels[s] for s in S]} (p={p:.4f})')

    # Step 4：循环结束后统一删除独立边
    for (i, j) in indep_edges:
        cg.remove_edge(i, j)
        # cg.remove_edge(j, i)

    return cg
