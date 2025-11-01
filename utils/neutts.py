import re
from pathlib import Path

import librosa
import numpy as np
import torch
from loguru import logger
from perth.perth_net.perth_net_implicit.perth_watermarker import PerthImplicitWatermarker
from phonemizer.backend import EspeakBackend
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.model import NeuCodec


class NeuTTSAir:
    def __init__(
            self,
            backbone_repo="models/neutts-air",
            backbone_device="cpu",
            codec_repo="models/neucodec",
            codec_device="cpu",
            language="cmn"
    ):
        assert language in ["cmn", "en-us", "yue"], "language必须是cmn、en-us、yue其中一个，cmn是普通话、en-us是英语，yue是粤语"
        self.sample_rate = 24_000
        self.max_context = 2048
        self.hop_length = 480
        self.streaming_overlap_frames = 1
        self.streaming_frames_per_chunk = 25
        self.streaming_lookforward = 5
        self.streaming_lookback = 50
        self.streaming_stride_samples = self.streaming_frames_per_chunk * self.hop_length

        # 加载水印器
        self.watermarker = PerthImplicitWatermarker()

        # HF 分词器
        self.tokenizer = None

        # 加载音素化器与模型
        logger.info("正在加载音素化器...")
        self.phonemizer = EspeakBackend(language=language, preserve_punctuation=True, with_stress=True)

        logger.info(f"正在从 {backbone_repo} 加载骨干模型，设备为 {backbone_device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_repo, local_files_only=True)
        self.backbone = AutoModelForCausalLM.from_pretrained(backbone_repo, local_files_only=True)
        self.backbone.to(torch.device(backbone_device))

        self.codec = NeuCodec.from_pretrained(codec_repo, local_files_only=True)
        self.codec.eval().to(codec_device)

    def infer(self, text: str, ref_codes: torch.Tensor, ref_text: str) -> np.ndarray:
        """
        使用 TTS 模型与参考音频，从文本生成语音。

        参数:
            text (str): 要转换为语音的输入文本。
            ref_codes (torch.tensor): 参考音频的编码。
            ref_text (str): 参考音频对应的文本。

        返回:
            np.ndarray: 生成的语音波形。
        """

        # 生成标记
        prompt_ids = self._apply_chat_template(ref_codes, ref_text, text)
        output_str = self._infer_torch(prompt_ids)

        # 解码
        wav = self._decode(output_str)
        watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=24_000)

        return watermarked_wav

    def encode_reference(self, ref_audio_path: str | Path):
        wav, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        with torch.no_grad():
            ref_codes = self.codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
        return ref_codes

    def _decode(self, codes: str):
        # 使用正则表达式提取语音 token 的 ID
        speech_ids = [int(num) for num in re.findall(r"<\|speech_(\d+)\|>", codes)]

        if len(speech_ids) > 0:
            with torch.no_grad():
                codes = torch.tensor(speech_ids, dtype=torch.long)[None, None, :].to(self.codec.device)
                recon = self.codec.decode_code(codes).cpu().numpy()

            return recon[0, 0, :]
        else:
            raise ValueError("没有找到有效的语音 token 在输出中。")

    def _to_phones(self, text: str) -> str:
        phones = self.phonemizer.phonemize([text])
        phones = phones[0].split()
        phones = " ".join(phones)
        return phones

    def _apply_chat_template(
            self, ref_codes: list[int], ref_text: str, input_text: str
    ) -> list[int]:

        input_text = self._to_phones(ref_text) + " " + self._to_phones(input_text)
        speech_replace = self.tokenizer.convert_tokens_to_ids("<|SPEECH_REPLACE|>")
        speech_gen_start = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
        text_replace = self.tokenizer.convert_tokens_to_ids("<|TEXT_REPLACE|>")
        text_prompt_start = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_START|>")
        text_prompt_end = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_END|>")

        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        chat = """user: Convert the text to speech:<|TEXT_REPLACE|>\nassistant:<|SPEECH_REPLACE|>"""
        ids = self.tokenizer.encode(chat)

        text_replace_idx = ids.index(text_replace)
        ids = (
                ids[:text_replace_idx]
                + [text_prompt_start]
                + input_ids
                + [text_prompt_end]
                + ids[text_replace_idx + 1:]  # noqa
        )

        speech_replace_idx = ids.index(speech_replace)
        codes_str = "".join([f"<|speech_{i}|>" for i in ref_codes])
        codes = self.tokenizer.encode(codes_str, add_special_tokens=False)
        ids = ids[:speech_replace_idx] + [speech_gen_start] + list(codes)

        return ids

    def _infer_torch(self, prompt_ids: list[int]) -> str:
        prompt_tensor = torch.tensor(prompt_ids).unsqueeze(0).to(self.backbone.device)
        speech_end_id = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
        with torch.no_grad():
            output_tokens = self.backbone.generate(
                prompt_tensor,
                max_length=self.max_context,
                eos_token_id=speech_end_id,
                do_sample=True,
                temperature=1.0,
                top_k=50,
                use_cache=True,
                min_new_tokens=50,
            )
        input_length = prompt_tensor.shape[-1]
        output_str = self.tokenizer.decode(output_tokens[0, input_length:].cpu().numpy().tolist(),
                                           add_special_tokens=False)
        return output_str
