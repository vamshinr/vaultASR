import json

with open("src/vernacula_cpp/models/whisper_tiny_en/vocab.json", "r", encoding="utf-8") as f:
    d = json.load(f)

inv = {v: k for k, v in d.items()}

lines = []
for i in range(max(inv.keys()) + 1):
    v = inv.get(i, "")
    v = v.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r")
    lines.append(f'"{v}"')

with open("src/vernacula_cpp/src/whisper_tokens.hpp", "w", encoding="utf-8") as f:
    f.write("#pragma once\n#include <string>\n#include <vector>\nconst std::vector<std::string> whisper_vocab = {\n")
    f.write(",\n".join(lines))
    f.write("\n};\n")
