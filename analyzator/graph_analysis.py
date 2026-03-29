from collections import Counter, defaultdict, deque

# имя ноды -> дочерние узлы
def _index_nodes_by_output(nodes):
    producer_by_tensor = {}

    for idx, node in enumerate(nodes):
        for output_name in node.get("outputs", []):
            producer_by_tensor[output_name] = idx

    return producer_by_tensor


# имя ноды -> узлы предшественники
def _index_nodes_by_input(nodes):
    consumers_by_tensor = defaultdict(list)

    for idx, node in enumerate(nodes):
        for input_name in node.get("inputs", []):
            consumers_by_tensor[input_name].append(idx)

    return dict(consumers_by_tensor)

# постройка графа
def _build_node_adjacency(nodes, producer_by_tensor):
    adjacency = defaultdict(set)

    for current_idx, node in enumerate(nodes):
        for input_name in node.get("inputs", []):
            producer_idx = producer_by_tensor.get(input_name)
            if producer_idx is not None and producer_idx != current_idx:
                adjacency[producer_idx].add(current_idx)

    adjacency = {k: sorted(v) for k, v in adjacency.items()}

    indegree = {idx: 0 for idx in range(len(nodes))}
    for src, dsts in adjacency.items():
        for dst in dsts:
            indegree[dst] += 1

    return adjacency, indegree

# топологическая сортировка графа + проверка на зацикленность
def _topological_sort(node_count, adjacency, indegree):
    indegree_copy = dict(indegree)
    queue = deque()

    for idx in range(node_count):
        if indegree_copy[idx] == 0:
            queue.append(idx)

    topo_order = []

    while queue:
        node_idx = queue.popleft()
        topo_order.append(node_idx)

        for neighbor in adjacency.get(node_idx, []):
            indegree_copy[neighbor] -= 1
            if indegree_copy[neighbor] == 0:
                queue.append(neighbor)

    has_cycle = len(topo_order) != node_count

    return topo_order, has_cycle

# глубина нод
def _estimate_node_depths(node_count, adjacency, topo_order):

    reverse_adj = defaultdict(list)

    for src, dst_list in adjacency.items():
        for dst in dst_list:
            reverse_adj[dst].append(src)

    depth_by_node = {idx: 0 for idx in range(node_count)}

    for node_idx in topo_order:
        parents = reverse_adj.get(node_idx, [])
        if not parents:
            depth_by_node[node_idx] = 0
        else:
            depth_by_node[node_idx] = 1 + max(depth_by_node[parent] for parent in parents)

    return depth_by_node


def analyze_graph(model_data):

    nodes = model_data.get("nodes", [])                         # список нод
    inputs = model_data.get("inputs", [])                       # входящие узлы
    outputs = model_data.get("outputs", [])                     # выходящие
    internal_tensors = model_data.get("internal_tensors", [])   # промежуточные
    initializers = model_data.get("initializers", [])           # веса

    
    # Количества
    node_count = len(nodes)
    input_count = len(inputs)
    output_count = len(outputs)
    internal_tensor_count = len(internal_tensors)
    initializer_count = len(initializers)

    # Счётчик операторов
    op_histogram = Counter(node.get("op_type", "UNKNOWN") for node in nodes)

    # Индексы входящик/выходящих узлов
    producer_by_tensor = _index_nodes_by_output(nodes)
    consumers_by_tensor = _index_nodes_by_input(nodes)

    # Граф и количество входящих нод
    adjacency, indegree = _build_node_adjacency(nodes, producer_by_tensor)

    # отсортированный и наличие циклов
    topo_order, has_cycle = _topological_sort(node_count, adjacency, indegree)

    # глубина нод
    depth_by_node = _estimate_node_depths(node_count, adjacency, topo_order)
    max_depth = max(depth_by_node.values()) if depth_by_node else 0

    # Поиск ветвящихся узлов
    branching_nodes = []
    for idx, node in enumerate(nodes):
        total_consumers = 0

        for output_name in node.get("outputs", []):
            total_consumers += len(consumers_by_tensor.get(output_name, []))

        if total_consumers > 1:
            branching_nodes.append({
                "node_index": idx,
                "node_name": node.get("name", ""),
                "op_type": node.get("op_type", ""),
                "consumer_count": total_consumers,
            })

    # Листовые узлы
    leaf_nodes = []
    for idx, node in enumerate(nodes):
        total_consumers = 0

        for output_name in node.get("outputs", []):
            total_consumers += len(consumers_by_tensor.get(output_name, []))

        if total_consumers == 0:
            leaf_nodes.append({
                "node_index": idx,
                "node_name": node.get("name", ""),
                "op_type": node.get("op_type", ""),
            })

    # Узлы - источники
    source_nodes = []
    for idx in range(node_count):
        if indegree.get(idx, 0) == 0:
            node = nodes[idx]
            source_nodes.append({
                "node_index": idx,
                "node_name": node.get("name", ""),
                "op_type": node.get("op_type", ""),
            })

    # Подробная информация по каждой ноде (для сравнения моделей)
    node_summaries = []

    for idx, node in enumerate(nodes):
        produced_tensors = node.get("outputs", [])
        consumed_by = []

        for output_name in produced_tensors:
            for consumer_idx in consumers_by_tensor.get(output_name, []):
                consumed_by.append(consumer_idx)

        node_summaries.append({
            "node_index": idx,                          # id
            "node_name": node.get("name", ""),          # имя
            "op_type": node.get("op_type", ""),         # тип операции
            "inputs": list(node.get("inputs", [])),     # список входных тензоров (в том числе веса)
            "outputs": list(node.get("outputs", [])),   # список выходных тензоров, которая создает эта нода
            "in_degree": indegree.get(idx, 0),          # количество входящих ребер (сколько нод подают данные)
            "out_degree": len(set(consumed_by)),        # сколько уникальных нод, использующие эту ноду
            "depth": depth_by_node.get(idx, 0),         # глубина
        })

    return {
        "node_count": node_count,                       # количество операций (нод)
        "input_count": input_count,                     # количество входных тензоров (только внешние входы, без весов)
        "output_count": output_count,                   # количество выходных тензоров (резльтатов)
        "internal_tensor_count": internal_tensor_count, # количество промежуточных тензоров (value_info)
        "initializer_count": initializer_count,         # количество параметровых тензоров
        "op_histogram": dict(op_histogram),             # тип операции и их количество
        "producer_by_tensor": producer_by_tensor,       # кто произвёл каждый тензор
        "consumers_by_tensor": consumers_by_tensor,     # кто использует тензор
        "adjacency": adjacency,                         # граф зависимостей
        "indegree": indegree,                           # сколько зависимостей у ноды от других нод
        "topological_order": topo_order,                # список индексов нод в порядке вычисления.
        "has_cycle": has_cycle,                         # есть ли цикл в графе
        "depth_by_node": depth_by_node,                 # глубина нод
        "max_depth": max_depth,                         # максимальная глубина графа
        "branching_nodes": branching_nodes,             # ноды, где граф разветвляется 
        "leaf_nodes": leaf_nodes,                       # ноды-листья. не обязательно output модели
        "source_nodes": source_nodes,                   # первая нода, где начинается вычисление
        "node_summaries": node_summaries,               # информация о нодах
    }