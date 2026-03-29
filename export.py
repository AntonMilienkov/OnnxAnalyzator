import torch
import torch.nn as nn
from modelBranch import BranchingModel

model = BranchingModel()

model.load_state_dict(torch.load("./project/PyTorchAnalyze/newModel.pt"))

dummy = torch.randn(1, 8)  # пример входа (batch=16, features=20)

torch.onnx.export(
    model,
    dummy,
    "./project/ONNXAnalyze/modelA.onnx",
    export_params=True,       
    opset_version=17,          
    do_constant_folding=True,  
    input_names=["input"],
    output_names=["output"],
)
print("saved modelA.onnx")

dummy2 = torch.randn(1, 8)  # пример входа (batch=16, features=20)

torch.onnx.export(
    model,
    dummy2,
    "./project/ONNXAnalyze/modelB.onnx",
    export_params=True,       
    opset_version=17,          
    do_constant_folding=True,  
    input_names=["input"],
    output_names=["output"],
)
print("saved modelB.onnx")