import onnx
model = onnx.load("models/wespeaker_pyannote.onnx")
print("Input shape:", [d.dim_value if d.dim_value > 0 else d.dim_param for d in model.graph.input[0].type.tensor_type.shape.dim])
