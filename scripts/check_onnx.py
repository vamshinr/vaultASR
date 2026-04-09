import onnx
model = onnx.load("models/wespeaker_pyannote.onnx")
print("Input:", model.graph.input[0].name)
print("Output:", model.graph.output[0].name)
