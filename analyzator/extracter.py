import onnx
from onnx import shape_inference, numpy_helper


def extract_raw_entities(model_path: str, run_shape_inference: bool = True) -> dict:
    model = onnx.load(model_path)

    # Проверка корректности модели
    onnx.checker.check_model(model)

    # Попытка восстановить формы промежуточных тензоров
    if run_shape_inference:
        try:
            model = shape_inference.infer_shapes(model)
        except Exception as e:
            print(f"[WARN] Shape inference не удалось выполнить: {e}")

    graph = model.graph



    # достает базовую информацию о ValueInfoProto в обычный словарь
    def raw_tensor_info(v):
        tensor_type = v.type.tensor_type

        # Числовой код типа данных ONNX
        elem_type = tensor_type.elem_type

        # shape.dim — список размерностей
        # если размерность неизвестна, ставим "?"
        shape = []
        for d in tensor_type.shape.dim:
            if d.dim_value > 0:
                shape.append(d.dim_value)
            elif d.dim_param:
                shape.append(d.dim_param)
            else:
                shape.append("?")

        return {
            "name": v.name,
            "elem_type": elem_type,
            "shape": shape,
        }

    # информация о TensorProto (веса)
    def raw_initializer_info(init):
        arr = numpy_helper.to_array(init)

        return {
            "name": init.name,
            "data_type": init.data_type,
            "shape": list(arr.shape),
            "values": arr,  # numpy-массив
        }

    # информация о NodeProto (узел графа / операция)
    def raw_node_info(node):
        attrs = {}
        for attr in node.attribute:
            attrs[attr.name] = onnx.helper.get_attribute_value(attr)

        return {
            "name": node.name,
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "attributes": attrs,
        }


    inputs = [raw_tensor_info(v) for v in graph.input]
    outputs = [raw_tensor_info(v) for v in graph.output]
    internal_tensors = [raw_tensor_info(v) for v in graph.value_info]
    nodes = [raw_node_info(node) for node in graph.node]
    initializers = [raw_initializer_info(init) for init in graph.initializer]

    # Возврат единым словарём
    return {
        "model": model,
        "inputs": inputs,
        "outputs": outputs,
        "internal_tensors": internal_tensors,
        "nodes": nodes,
        "initializers": initializers,
    }

