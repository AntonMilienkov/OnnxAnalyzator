import numpy as np

# перевод многомерного массива в линейный
def _flatten_array(arr):
    return arr.reshape(-1)

# внутренняя функция для анализа весового тензора
def _compute_weight_stats(weight):
    arr = weight["values"]

    arr64 = arr.astype(np.float64, copy=False)
    flat = _flatten_array(arr64)

    nan_count = int(np.isnan(flat).sum())
    posinf_count = int(np.isposinf(flat).sum())
    neginf_count = int(np.isneginf(flat).sum())
    inf_count = posinf_count + neginf_count

    stats = {
        "name": weight["name"],                 # имя
        "shape": list(arr.shape),               # форма
        "dtype": str(arr.dtype),                # тип данных
        "size": int(arr.size),                  # количество элементов в тензоре
        "min": None,                            # минимальное конечное значение в тензоре
        "max": None,                            # максимальное конечное значение в тензоре
        "mean": None,                           # среднее значение (конечных) в тензоре
        "std": None,                            # среднее отклонение
        "l1_norm": None,                        # L1-норма: сумма модулей всех конечных значений
        "l2_norm": None,                        # L2-норма: евклидова длина вектора весов
        "nan_count": nan_count,                 # сколько значений NaN (говорит об ошибках)
        "inf_count": inf_count,                 # сколько сколько +-Inf
        "all_zero": bool(np.all(arr == 0)),     # True, если все элементы тензора равны нулю
    }


    # для подсчета характеристик рассмаотрим только конечные значения
    finite_mask = np.isfinite(flat)
    finite_values = flat[finite_mask]

    if finite_values.size > 0:
        stats["min"] = float(np.min(finite_values))
        stats["max"] = float(np.max(finite_values))
        stats["mean"] = float(np.mean(finite_values))
        stats["std"] = float(np.std(finite_values))
        stats["l1_norm"] = float(np.sum(np.abs(finite_values)))
        stats["l2_norm"] = float(np.linalg.norm(finite_values))     

    return stats

# имя_веса -> весовой тензор
def index_weights_by_name(model_data):
    result = {}

    for weight in model_data.get("initializers", []):
        result[weight["name"]] = weight

    return result

# Анализ весов модели
def analyze_weights(model_data):
    weights = model_data.get("initializers", [])

    stats_by_weight = []
    total_parameter_count = 0
    weights_with_nan = []
    weights_with_inf = []
    zero_weights = []

    for weight in weights:
        stats = _compute_weight_stats(weight)
        stats_by_weight.append(stats)

        total_parameter_count += stats["size"]

        if stats["nan_count"] > 0:
            weights_with_nan.append(weight["name"])

        if stats["inf_count"] > 0:
            weights_with_inf.append(weight["name"])

        if stats["all_zero"]:
            zero_weights.append(weight["name"])

    return {
        "weight_count": len(weights),                   # количество весовых тензоров (initializers)
        "total_parameter_count": total_parameter_count, # исуммарное количество всех параметров модели
        "weights_with_nan": weights_with_nan,           # имена весов, содержащих NaN
        "weights_with_inf": weights_with_inf,           # имена весов, содержащих бесконечные значения
        "zero_weights": zero_weights,                   # имена весов, полностью состоящих из нулей
        "weights": stats_by_weight,                     # детальные статистики по каждому весовому тензору
    }
