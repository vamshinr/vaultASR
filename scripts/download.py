from modelscope.hub.file_download import model_file_download
import shutil

print("Downloading CampPlus ONNX from ModelScope...")
try:
    path = model_file_download('damo/speech_campplus_sv_en_voxceleb_16k', 'campplus.onnx')
    shutil.copy(path, "models/wespeaker_pyannote.onnx")
    print("CampPlus ONNX downloaded successfully via ModelScope!")
except Exception as e:
    print("CampPlus Failed:", e)
