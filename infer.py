from utils.neutts import NeuTTSAir
import soundfile as sf


tts = NeuTTSAir(
   backbone_repo="./models/neutts-air-zh",
   backbone_device="cpu",
   codec_repo="./models/neucodec",
   codec_device="cpu",
   language="cmn"
)
input_text = "你们好，我是夜雨飘零，很高兴见到你。"

ref_text = "近几年，不但我用书给女儿压岁，也劝说亲朋不要给女儿压岁钱，而改送压岁书。"
ref_audio_path = "data/test.wav"

ref_codes = tts.encode_reference(ref_audio_path)
wav = tts.infer(input_text, ref_codes, ref_text)

sf.write("output.wav", wav, 24000)
