def pc_with_fnode(normal_df, anomalous_df, alpha, bins=None,
                  localized=False, verbose=VERBOSE):
    data = _preprocess_for_fnode(normal_df, anomalous_df, bins)
    cg = run_pc(data, alpha, localized=localized, verbose=verbose)
    return cg.nx_graph

def save_graph(graph, file):
    nx.draw_networkx(graph)
    plt.savefig(file)