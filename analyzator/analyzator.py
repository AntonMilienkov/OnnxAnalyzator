from extracter import extract_raw_entities
from graph_analysis import analyze_graph
from pretty_print import pretty_print_dict
from compare import summarize_graph_comparison, compare_graphs
import onnx
from collections import Counter


extracted_model_a = extract_raw_entities("./OnnxModels/model_A.onnx")
extracted_model_b = extract_raw_entities("./OnnxModels/model_B.onnx")
extracted_model_bb = extract_raw_entities("./OnnxModels/model_BB.onnx")
extracted_model_c = extract_raw_entities("./OnnxModels/model_C.onnx")
extracted_model_d = extract_raw_entities("./OnnxModels/model_D.onnx")

print("================================================================================================================")
print("                     A   &   B        ")
print("================================================================================================================")

res = summarize_graph_comparison(compare_graphs(analyze_graph(extracted_model_a), analyze_graph(extracted_model_b)))

pretty_print_dict(res)


print("================================================================================================================")
print("                     A   &   BB        ")
print("================================================================================================================")
res = summarize_graph_comparison(compare_graphs(analyze_graph(extracted_model_a), analyze_graph(extracted_model_bb)))

pretty_print_dict(res, True)


print("================================================================================================================")
print("                     A   &   C        ")
print("================================================================================================================")
res = summarize_graph_comparison(compare_graphs(analyze_graph(extracted_model_a), analyze_graph(extracted_model_c)))

pretty_print_dict(res, True)



print("================================================================================================================")
print("                     A   &   D        ")
print("================================================================================================================")
res = summarize_graph_comparison(compare_graphs(analyze_graph(extracted_model_a), analyze_graph(extracted_model_d)))

pretty_print_dict(res, True)



def show_onnx_ops(path):
    model = onnx.load(path)
    ops = [node.op_type for node in model.graph.node]
    print(path)
    print("node_count =", len(ops))
    print("ops =", ops)
    print("hist =", Counter(ops))
    print()

show_onnx_ops("./OnnxModels/model_A.onnx")
show_onnx_ops("./OnnxModels/model_B.onnx")
show_onnx_ops("./OnnxModels/model_BB.onnx")