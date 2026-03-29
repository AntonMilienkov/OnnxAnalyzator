import numpy as np
from collections import Counter

# =========================
# Сравнение весов
# =========================

# перевод многомерного массива в линейный
def _flatten_array(arr):
    return arr.reshape(-1)

# косинусовое сходство
def _cosine_similarity(a, b):
    a_flat = _flatten_array(a).astype(np.float64)
    b_flat = _flatten_array(b).astype(np.float64)

    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)

    if norm_a == 0.0 or norm_b == 0.0:
        return None

    cos = np.dot(a_flat, b_flat) / (norm_a * norm_b)
    return float(cos)


def _compare_two_weights(weight_a, weight_b, atol=1e-8, rtol=1e-5):
    arr_a = weight_a["values"]
    arr_b = weight_b["values"]

    result = {
        "name_a": weight_a["name"],               # имя сравниваемого весового тензора
        "name_b": weight_b["name"],               # имя сравниваемого второго весового тензора
        "status": "ok",                           # ok / shape_mismatch / dtype_mismatch / non_finite_values
        "shape_a": list(arr_a.shape),             # форма веса в модели A
        "shape_b": list(arr_b.shape),             # форма веса в модели B
        "dtype_a": str(arr_a.dtype),              # тип данных веса в модели A
        "dtype_b": str(arr_b.dtype),              # тип данных веса в модели B
        "exact_equal": None,                      # точное совпадение значений
        "allclose": None,                         # близость с учетом atol/rtol
        "max_abs_diff": None,                     # максимальное абсолютное отличие
        "mean_abs_diff": None,                    # среднее абсолютное отличие
        "rmse": None,                             # среднеквадратическая ошибка
        "cosine_similarity": None,                # косинусное сходство
        "nan_count_a": int(np.isnan(arr_a).sum()),# количество NaN в A
        "nan_count_b": int(np.isnan(arr_b).sum()),# количество NaN в B
        "inf_count_a": int(np.isinf(arr_a).sum()),# количество Inf в A
        "inf_count_b": int(np.isinf(arr_b).sum()),# количество Inf в B
    }


    if arr_a.shape != arr_b.shape:
        result["status"] = "shape_mismatch"
        return result

    if arr_a.dtype != arr_b.dtype:
        result["status"] = "dtype_mismatch"

    a64 = arr_a.astype(np.float64, copy=False)
    b64 = arr_b.astype(np.float64, copy=False)

    if not np.all(np.isfinite(a64)) or not np.all(np.isfinite(b64)):
        if result["status"] == "ok":
            result["status"] = "non_finite_values"
        return result

    diff = a64 - b64
    abs_diff = np.abs(diff)

    result["exact_equal"] = bool(np.array_equal(arr_a, arr_b))
    result["allclose"] = bool(np.allclose(a64, b64, atol=atol, rtol=rtol))
    result["max_abs_diff"] = float(np.max(abs_diff))
    result["mean_abs_diff"] = float(np.mean(abs_diff))
    result["rmse"] = float(np.sqrt(np.mean(diff ** 2)))
    result["cosine_similarity"] = _cosine_similarity(a64, b64)

    return result

# имя_веса -> весовой тензор
def index_weights_by_name(model_data):
    result = {}
    for weight in model_data.get("initializers", []):
        result[weight["name"]] = weight
    return result

# Сравнение весов моделей
def compare_weights(model_data_a, model_data_b, atol=1e-8, rtol=1e-5):
    weights_a = index_weights_by_name(model_data_a)
    weights_b = index_weights_by_name(model_data_b)

    names_a = set(weights_a.keys())
    names_b = set(weights_b.keys())

    only_in_a = sorted(names_a - names_b)
    only_in_b = sorted(names_b - names_a)
    common_names = sorted(names_a & names_b)

    comparisons = []

    exact_equal_count = 0
    allclose_count = 0
    shape_mismatch_count = 0
    dtype_mismatch_count = 0
    non_finite_count = 0
    numerically_different_count = 0

    for name in common_names:
        cmp_result = _compare_two_weights(
            weights_a[name],
            weights_b[name],
            atol=atol,
            rtol=rtol,
        )
        comparisons.append(cmp_result)

        if cmp_result["status"] == "shape_mismatch":
            shape_mismatch_count += 1
            continue

        if cmp_result["status"] == "dtype_mismatch":
            dtype_mismatch_count += 1

        if cmp_result["status"] == "non_finite_values":
            non_finite_count += 1
            continue

        if cmp_result["exact_equal"]:
            exact_equal_count += 1

        if cmp_result["allclose"]:
            allclose_count += 1
        else:
            numerically_different_count += 1

    return {
        "only_in_a": only_in_a,                                     # имена весов, присутствующих только в модели A
        "only_in_b": only_in_b,                                     # имена весов, присутствующих только в модели B
        "common_weight_count": len(common_names),                   # число общих весов
        "exact_equal_count": exact_equal_count,                     # число точно совпавших весов
        "allclose_count": allclose_count,                           # число численно близких весов
        "shape_mismatch_count": shape_mismatch_count,               # число весов с разной формой
        "dtype_mismatch_count": dtype_mismatch_count,               # число весов с разным dtype
        "non_finite_count": non_finite_count,                       # число сравнений, где есть NaN/Inf
        "numerically_different_count": numerically_different_count, # число весов, не прошедших allclose
        "comparisons": comparisons,                                 # подробное сравнение по каждому общему весу
        "tolerance": {
            "atol": atol,                                           # абсолютный допуск
            "rtol": rtol,                                           # относительный допуск
        },
    }

# Более сжатый вывод
def summarize_weight_comparison(weight_diff_report):
    return {
        "only_in_a_count": len(weight_diff_report.get("only_in_a", [])),
        "only_in_b_count": len(weight_diff_report.get("only_in_b", [])),
        "common_weight_count": weight_diff_report.get("common_weight_count", 0),
        "exact_equal_count": weight_diff_report.get("exact_equal_count", 0),
        "allclose_count": weight_diff_report.get("allclose_count", 0),
        "shape_mismatch_count": weight_diff_report.get("shape_mismatch_count", 0),
        "dtype_mismatch_count": weight_diff_report.get("dtype_mismatch_count", 0),
        "non_finite_count": weight_diff_report.get("non_finite_count", 0),
        "numerically_different_count": weight_diff_report.get("numerically_different_count", 0),
    }


# =========================
# Сравнение графа
# =========================

# Возвращет ключ ноды: имя или сигнатуру
def _node_key(node_summary):
    """
    Ключ для сопоставления нод между моделями.
    Приоритет: node_name.
    Если имени нет — используем структурный fallback.
    """
    node_name = node_summary.get("node_name", "")
    if node_name:
        return ("name", node_name)

    return (
        "signature",
        node_summary.get("op_type", ""),
        tuple(node_summary.get("inputs", [])),
        tuple(node_summary.get("outputs", [])),
    )

# словарь ключ -> нода
def _index_node_summaries(node_summaries):
    result = {}
    for node in node_summaries:
        key = _node_key(node)
        result[key] = node
    return result

# Сравнение частот операций
def _compare_histograms(hist_a, hist_b):
    keys = sorted(set(hist_a.keys()) | set(hist_b.keys()))
    diff = []

    for key in keys:
        count_a = hist_a.get(key, 0)
        count_b = hist_b.get(key, 0)
        if count_a != count_b:
            diff.append({
                "op_type": key,             # тип операции
                "count_a": count_a,         # количество в модели A
                "count_b": count_b,         # количество в модели B
                "delta": count_a - count_b, # разница A - B
            })
        

    return diff


def _compare_two_nodes(node_a, node_b):
    result = {
        "node_key_a": _node_key(node_a),                                                    # ключ сопоставления ноды
        "node_key_b": _node_key(node_b),                                                    # ключ сопоставления ноды
        "node_name_a": node_a.get("node_name", ""),                                         # имя ноды в модели A (может совпадать с ключом)
        "node_name_b": node_b.get("node_name", ""),                                         # имя ноды в модели B
        "op_type_a": node_a.get("op_type", ""),                                             # тип операции в модели A
        "op_type_b": node_b.get("op_type", ""),                                             # тип операции в модели B
        "same_op_type": node_a.get("op_type", "") == node_b.get("op_type", ""),             # проверка на соответсвие операции
        "same_inputs": list(node_a.get("inputs", [])) == list(node_b.get("inputs", [])),    # проверка на соответсвие входов
        "same_outputs": list(node_a.get("outputs", [])) == list(node_b.get("outputs", [])), # проверка на соответсвие выходов
        "same_in_degree": node_a.get("in_degree", 0) == node_b.get("in_degree", 0),         # проверка на соответсвие входящих ребер
        "same_out_degree": node_a.get("out_degree", 0) == node_b.get("out_degree", 0),      # проверка на соответсвие выходящих ребер
        "same_depth": node_a.get("depth", 0) == node_b.get("depth", 0),                     # проверка на соответсвие глубины
        "inputs_a": list(node_a.get("inputs", [])),                                         # список входов в ноду модели A
        "inputs_b": list(node_b.get("inputs", [])),                                         # список входов в ноду модели B
        "outputs_a": list(node_a.get("outputs", [])),                                       # список выходов из ноды модели A
        "outputs_b": list(node_b.get("outputs", [])),                                       # список выходов из ноды модели B
        "in_degree_a": node_a.get("in_degree", 0),                                          # входящие ребра модели A
        "in_degree_b": node_b.get("in_degree", 0),                                          # входящие ребра модели B
        "out_degree_a": node_a.get("out_degree", 0),                                        # выходящие ребра модели A
        "out_degree_b": node_b.get("out_degree", 0),                                        # выходящие ребра модели B
        "depth_a": node_a.get("depth", 0),                                                  # глубина ноды A
        "depth_b": node_b.get("depth", 0),                                                  # глубина ноды B
    }

    result["is_fully_equal"] = all([                                                        # являются ли ноды идентичными
        result["same_op_type"],
        result["same_inputs"],
        result["same_outputs"],
        result["same_in_degree"],
        result["same_out_degree"],
        result["same_depth"],
    ])

    return result

# вход: результаты analyze_graph
def compare_graphs(graph_report_a, graph_report_b):
    global_stats = {                                                                
        "node_count_a": graph_report_a.get("node_count", 0),
        "node_count_b": graph_report_b.get("node_count", 0),

        "input_count_a": graph_report_a.get("input_count", 0),
        "input_count_b": graph_report_b.get("input_count", 0),

        "output_count_a": graph_report_a.get("output_count", 0),
        "output_count_b": graph_report_b.get("output_count", 0),

        "internal_tensor_count_a": graph_report_a.get("internal_tensor_count", 0),
        "internal_tensor_count_b": graph_report_b.get("internal_tensor_count", 0),

        "initializer_count_a": graph_report_a.get("initializer_count", 0),
        "initializer_count_b": graph_report_b.get("initializer_count", 0),

        "max_depth_a": graph_report_a.get("max_depth", 0),
        "max_depth_b": graph_report_b.get("max_depth", 0),

        "has_cycle_a": graph_report_a.get("has_cycle", False),
        "has_cycle_b": graph_report_b.get("has_cycle", False),
    }

    op_histogram_a = graph_report_a.get("op_histogram", {})
    op_histogram_b = graph_report_b.get("op_histogram", {})
    op_histogram_diff = _compare_histograms(op_histogram_a, op_histogram_b)

    node_summaries_a = graph_report_a.get("node_summaries", [])
    node_summaries_b = graph_report_b.get("node_summaries", [])

    indexed_a = _index_node_summaries(node_summaries_a)
    indexed_b = _index_node_summaries(node_summaries_b)

    keys_a = set(indexed_a.keys())
    keys_b = set(indexed_b.keys())

    only_in_a = sorted(keys_a - keys_b, key=str)
    only_in_b = sorted(keys_b - keys_a, key=str)
    common_keys = sorted(keys_a & keys_b, key=str)

    node_comparisons = []

    fully_equal_node_count = 0
    changed_node_count = 0
    op_type_changed_count = 0
    io_changed_count = 0
    topology_changed_count = 0

    for key in common_keys:
        cmp_result = _compare_two_nodes(indexed_a[key], indexed_b[key])
        node_comparisons.append(cmp_result)

        if cmp_result["is_fully_equal"]:
            fully_equal_node_count += 1
        else:
            changed_node_count += 1

        if not cmp_result["same_op_type"]:
            op_type_changed_count += 1

        if (not cmp_result["same_inputs"]) or (not cmp_result["same_outputs"]):
            io_changed_count += 1

        if (
            (not cmp_result["same_in_degree"]) or
            (not cmp_result["same_out_degree"]) or
            (not cmp_result["same_depth"])
        ):
            topology_changed_count += 1

    return {
        "global_stats": global_stats,                       # сравнение общих характеристик графа (различные количества, макс. глубина, наличие зацикленности)
        "op_histogram_diff": op_histogram_diff,             # различия по количеству операторов каждого типа
        "only_in_a": only_in_a,                             # ключи нод, присутствующих только в модели A
        "only_in_b": only_in_b,                             # ключи нод, присутствующих только в модели B
        "common_node_count": len(common_keys),              # число общих нод
        "fully_equal_node_count": fully_equal_node_count,   # число полностью совпавших нод
        "changed_node_count": changed_node_count,           # число нод с хотя бы одним отличием
        "op_type_changed_count": op_type_changed_count,     # число нод с изменившимся типом операции
        "io_changed_count": io_changed_count,               # число нод с изменившимися входами или выходами
        "topology_changed_count": topology_changed_count,   # число нод с изменением топологии
        "node_comparisons": node_comparisons,               # детальное сравнение нод
    }

# более сжатый результат анализа
def summarize_graph_comparison(graph_diff_report):
    global_stats = graph_diff_report.get("global_stats", {})
    op_histogram_diff = graph_diff_report.get("op_histogram_diff", [])

    return {
        "node_count_a": global_stats.get("node_count_a", 0),
        "node_count_b": global_stats.get("node_count_b", 0),
        "max_depth_a": global_stats.get("max_depth_a", 0),
        "max_depth_b": global_stats.get("max_depth_b", 0),
        "has_cycle_a": global_stats.get("has_cycle_a", False),
        "has_cycle_b": global_stats.get("has_cycle_b", False),
        "op_histogram_changed": len(op_histogram_diff) > 0,
        "only_in_a_count": len(graph_diff_report.get("only_in_a", [])),
        "only_in_b_count": len(graph_diff_report.get("only_in_b", [])),
        "common_node_count": graph_diff_report.get("common_node_count", 0),
        "fully_equal_node_count": graph_diff_report.get("fully_equal_node_count", 0),
        "changed_node_count": graph_diff_report.get("changed_node_count", 0),
        "op_type_changed_count": graph_diff_report.get("op_type_changed_count", 0),
        "io_changed_count": graph_diff_report.get("io_changed_count", 0),
        "topology_changed_count": graph_diff_report.get("topology_changed_count", 0),
    }


# ================================================
# Полное сравнение моделей (веса + граф)
# ================================================

def compare_models(model_data_a, model_data_b, graph_report_a, graph_report_b, atol=1e-8, rtol=1e-5):
    weight_diff = compare_weights(model_data_a, model_data_b, atol=atol, rtol=rtol)
    graph_diff = compare_graphs(graph_report_a, graph_report_b)

    return {
        "weights": weight_diff,                            # подробное сравнение весов
        "graph": graph_diff,                               # подробное сравнение графа
        "summary": {
            "weights": summarize_weight_comparison(weight_diff),
            "graph": summarize_graph_comparison(graph_diff),
        },
    }


def summarize_model_comparison(model_diff_report):
    return {
        "weights": model_diff_report.get("summary", {}).get("weights", {}),
        "graph": model_diff_report.get("summary", {}).get("graph", {}),
    }