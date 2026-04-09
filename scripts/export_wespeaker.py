import os
import torch
import torchaudio

os.makedirs("models", exist_ok=True)
token = os.environ.get("HUGGING_FACE_HUB_TOKEN")

# Allow Pyannote unsafe loading due to older checkpoint format
try:
    torch.serialization.add_safe_globals([getattr(torch, "torch_version", type(torch.__version__))])
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
except:
    pass

import warnings
warnings.filterwarnings("ignore")

try:
    from pyannote.audio import Model
    print("Loading Pyannote Model...")
    # This bypass avoids the WeightsUnpickler error directly
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM", use_auth_token=token)
    model.eval()
    
    print("Exporting Pyannote model to ONNX...")
    dummy_input = torch.randn(1, 1, 300, 80)
    torch.onnx.export(model, dummy_input, "models/wespeaker_pyannote.onnx", 
                      input_names=["fbank"], output_names=["embs"],
                      dynamic_axes={"fbank": {0: "batch_size", 2: "num_frames"}, "embs": {0: "batch_size"}})
    print("Successfully exported wespeaker_pyannote.onnx!")
except Exception as e:
    print(f"Failed to use pyannote: {e}")
