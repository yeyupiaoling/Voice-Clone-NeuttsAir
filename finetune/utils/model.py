import os.path
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
import torchaudio
from huggingface_hub import PyTorchModelHubMixin
from neucodec.codec_decoder_vocos import CodecDecoderVocos
from neucodec.codec_encoder import CodecEncoder
from neucodec.module import SemanticEncoder
from torchaudio import transforms as T
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel


class NeuCodec(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/neuphonic/neucodec",
    license="apache-2.0",
):

    def __init__(self, sample_rate: int, hop_length: int):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.semantic_model = Wav2Vec2BertModel.from_pretrained("../models/w2v-bert-2.0",
                                                                output_hidden_states=True,
                                                                local_files_only=True)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("../models/w2v-bert-2.0",
                                                                      local_files_only=True)
        self.SemanticEncoder_module = SemanticEncoder(1024, 1024, 1024)
        self.CodecEnc = CodecEncoder()
        self.generator = CodecDecoderVocos(hop_length=hop_length)
        self.fc_prior = nn.Linear(2048, 2048)
        self.fc_post_a = nn.Linear(2048, 1024)

    @property
    def device(self):
        return next(self.parameters()).device

    @classmethod
    def _from_pretrained(
            cls,
            *,
            model_id: str,
            revision: Optional[str] = None,
            cache_dir: Optional[str] = None,
            force_download: bool = False,
            proxies: Optional[Dict] = None,
            resume_download: bool = False,
            local_files_only: bool = False,
            token: Optional[str] = None,
            map_location: str = "cpu",
            strict: bool = True,
            **model_kwargs,
    ):

        if model_id.split("/")[-1] == "neucodec":
            ignore_keys = ["fc_post_s", "SemanticDecoder"]
        elif model_id.split("/")[-1] == "distill-neucodec":
            ignore_keys = []

        # initialize model
        model = cls(24_000, 480)

        # load weights
        state_dict = torch.load(os.path.join(model_id, "pytorch_model.bin"), map_location)
        contains_list = lambda s, l: any(i in s for i in l)
        state_dict = {
            k: v for k, v in state_dict.items()
            if not contains_list(k, ignore_keys)
        }

        model.load_state_dict(state_dict, strict=False)

        return model

    def _prepare_audio(self, audio_or_path: torch.Tensor | Path | str):

        # load from file
        if isinstance(audio_or_path, (Path, str)):
            y, sr = torchaudio.load(audio_or_path)
            if sr != 16_000:
                y, sr = (T.Resample(sr, 16_000)(y), 16_000)
                y = y[None, :]  # [1, T] -> [B, 1, T]

        # ensure input tensor is of correct shape
        elif isinstance(audio_or_path, torch.Tensor):
            y = audio_or_path
            if len(y.shape) == 3:
                y = audio_or_path
            else:
                raise ValueError(
                    f"NeuCodec expects tensor audio input to be of shape [B, 1, T] -- received shape: {y.shape}"
                )

        # pad audio
        pad_for_wav = 320 - (y.shape[-1] % 320)
        y = torch.nn.functional.pad(y, (0, pad_for_wav))

        return y

    def encode_code(self, audio_or_path: torch.Tensor | Path | str) -> torch.Tensor:
        """
        Args:
            audio_or_path: torch.Tensor [B, 1, T] | Path | str, input audio

        Returns:
            fsq_codes: torch.Tensor [B, 1, F], 50hz FSQ codes
        """

        # prepare inputs
        y = self._prepare_audio(audio_or_path)
        semantic_features = self.feature_extractor(
            y.squeeze(0), sampling_rate=16_000, return_tensors="pt"
        ).input_features.to(self.device)

        # acoustic encoding
        acoustic_emb = self.CodecEnc(y.to(self.device))
        acoustic_emb = acoustic_emb.transpose(1, 2)

        # semantic encoding
        semantic_output = (
            self.semantic_model(semantic_features).hidden_states[16].transpose(1, 2)
        )
        semantic_encoded = self.SemanticEncoder_module(semantic_output)

        # concatenate embeddings
        if acoustic_emb.shape[-1] != semantic_encoded.shape[-1]:
            min_len = min(acoustic_emb.shape[-1], semantic_encoded.shape[-1])
            acoustic_emb = acoustic_emb[:, :, :min_len]
            semantic_encoded = semantic_encoded[:, :, :min_len]
        concat_emb = torch.cat([semantic_encoded, acoustic_emb], dim=1)
        concat_emb = self.fc_prior(concat_emb.transpose(1, 2)).transpose(1, 2)

        # quantize
        _, fsq_codes, _ = self.generator(concat_emb, vq=True)
        return fsq_codes

    def decode_code(self, fsq_codes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fsq_codes: torch.Tensor [B, 1, F], 50hz FSQ codes

        Returns:
            recon: torch.Tensor [B, 1, T], reconstructed 24kHz audio
        """

        fsq_post_emb = self.generator.quantizer.get_output_from_indices(fsq_codes.transpose(1, 2))
        fsq_post_emb = fsq_post_emb.transpose(1, 2)
        fsq_post_emb = self.fc_post_a(fsq_post_emb.transpose(1, 2)).transpose(1, 2)
        recon = self.generator(fsq_post_emb.transpose(1, 2), vq=False)[0]
        return recon
