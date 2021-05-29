def traversal(x, nodes, stats):
    node = nodes[x]
    if node.leftChild != -1:
        traversal(node.leftChild, nodes, stats)
    if node.rightChild != -1:
        traversal(node.rightChild, nodes, stats)
    if node.leftChild == -1 and node.rightChild == -1:
        stats.append(node.impurityStats)

def decision_tree_parse(dt, sc, model_path):
    """
    dt: DecisionTreeClassificationModel
    spark: SparkSession
    model_path: model path on hdfs
    """
    dt.write().overwrite().save(model_path)
    nodeData = sc.read.parquet(model_path + '/data/')
    
    nodes = nodeData.collect()
    nodes = sorted(nodes, key=lambda x: x.id)
    
    stats = []
    traversal(0, nodes, stats)

    tree = []
    idx = 0
    for i in dt.toDebugString.split('\n'):
        if 'Pred' in i:
            tree.append(i + f': ({stats[idx]})')
            idx += 1
        else:
            tree.append(i)
    return tree