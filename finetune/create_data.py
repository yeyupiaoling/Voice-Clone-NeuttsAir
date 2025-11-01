import json
import os
import warnings

import torch
import torchaudio
from utils.model import NeuCodec
from torchaudio import transforms as T
from tqdm import tqdm

warnings.filterwarnings("ignore")

# 加载音频编码模型
model = NeuCodec.from_pretrained("../models/neucodec")
model.eval().cuda()

# 需要提前下载data_thchs30数据集，并将其解压到dataset目录下
data_dir = "dataset/data_thchs30/data"
save_data_path = "dataset/train_data.jsonl"

exists_data = set()
if os.path.exists(save_data_path):
    with open(save_data_path, "r", encoding="utf-8") as f_write:
        for line in f_write.readlines():
            data = json.loads(line)
            exists_data.add(data["audio_path"])

with open(save_data_path, "a", encoding="utf-8") as f_write:
    all_data = []
    # 遍历data_dir下的所有文件
    for file in tqdm(os.listdir(data_dir), desc="正在处理文件"):
        if '.trn' in file:
            file = os.path.join(data_dir, file).replace('\\', '/')
            with open(file, 'r', encoding='utf-8') as f:
                text = f.readline()
                text = ''.join(text.split())
                audio_path = file.replace('.trn', '')
                all_data.append((audio_path, text))

    # 处理数据
    for audio_path, text in tqdm(all_data, desc="正在处理数据"):
        if audio_path in exists_data:
            continue

        y, sr = torchaudio.load(audio_path)

        if sr != 16_000:
            y = T.Resample(sr, 16_000)(y)[None, ...]

        with torch.no_grad():
            if y.dim() == 2:
                y = y[None, ...]
            fsq_codes = model.encode_code(y).squeeze()
            f_write.write(json.dumps({"audio_path": audio_path,
                                      "duration": round(y.shape[-1] / 16_000, 3),
                                      "text": text,
                                      "codes": fsq_codes.tolist()}, ensure_ascii=False) + "\n")
