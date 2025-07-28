from ultralytics import YOLO

model = YOLO(r"C:\Users\lenovo\Programming\cnic_detection\runs\tiny\best.onnx")

class_names = model.names
print(class_names)

import onnx
from onnx import helper

model = onnx.load(r"C:\Users\lenovo\Programming\cnic_detection\runs\tiny\best.onnx")

graph = model.graph

print("Inputs:")

for input in graph.input:
    print(f"Name: {input.name}")
    for dim in input.type.tensor_type.shape.dim:
        print(f"Dimension:{dim.dim_param or dim.dim_value}")
    print(f"Type: {helper.mapping.TENSOR_TYPE_TO_NP_TYPE[input.type.tensor_type.elem_type]}")

print("Outputs:")
for output in graph.output:
    print(f"Name: {output.name}")
    for dim in output.type.tensor_type.shape.dim:
            print(f"Dimension:{dim.dim_param or dim.dim_value}")
    print(f"Type: {helper.mapping.TENSOR_TYPE_TO_NP_TYPE[input.type.tensor_type.elem_type]}")

import onnxruntime as ort

session = ort.InferenceSession(r"C:\Users\lenovo\Programming\cnic_detection\runs\tiny\best.onnx")

print("Inputs:")
for input in session.get_inputs():
    print(f"Name: {input.name}")
    print(f"Shape: {input.shape}")
    print(f"Type: {input.type}")

print("Outputs:")
for output in session.get_outputs():
    print(f"Name: {output.name}")
    print(f"Shape: {output.shape}")
    print(f"Type: {output.type}")