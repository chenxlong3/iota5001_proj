import networkx as nx

proxy_metrics = ["pagerank", "outdegree", "betweenness"]

def read_mtx(path,
    skip = 0,
    comments="#",
    delimiter=None,
    create_using=None,
    nodetype=None,
    data=True,
    edgetype=None,
    encoding="utf-8",):
    """
    Inputs:
        path: file path
        skip: the number of lines that should be skipped
    Outputs:
        networkx graph
    """
    with open(path, 'r') as f:
        str_list = f.readlines()
    return nx.parse_edgelist(
        str_list[skip:],
        comments=comments,
        delimiter=delimiter,
        create_using=create_using,
        nodetype=nodetype,
        data=data,
    )

def get_num_col(file_name):
    with open(file_name, "rb") as f:
        line = next(f)
        return len(line.decode("utf-8").split(' '))
