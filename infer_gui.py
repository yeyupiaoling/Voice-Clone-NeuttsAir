import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import wave

import pyaudio
import soundfile as sf
import torch
import winsound
from loguru import logger

from utils.neutts import NeuTTSAir


class TTSApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("夜雨飘零语音克隆合成")
        self._center_window(820, 560)

        # 模型路径，可以修改不同模型以支持不同语言
        self.model_dir = "./finetune/output"
        self.language = "cmn"

        # UI state
        self.reference_audio_path: str | None = None
        self.last_output_path: str | None = None
        self.tts_model: NeuTTSAir | None = None
        self.model_lock = threading.Lock()
        self.model_loaded: bool = False
        # 要运行的设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 录音相关状态
        self.is_recording: bool = False
        self.recording_thread: threading.Thread | None = None
        self.audio_stream = None
        self.frames = []

        # Styles
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TButton", padding=8)
        style.configure("TLabel", padding=4)
        style.configure("TEntry", padding=4)
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 14, "bold"))

        container = ttk.Frame(self.root, padding=16)
        container.pack(fill=tk.BOTH, expand=True)

        # Title
        ttk.Label(container, text="语音克隆与合成", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 8))

        # Reference selection frame
        ref_frame = ttk.LabelFrame(container, text="参考音频与对应文本", padding=12)
        ref_frame.pack(fill=tk.X, pady=8)

        # Audio file row
        audio_row = ttk.Frame(ref_frame)
        audio_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(audio_row, text="要克隆的音频 (WAV)：").pack(side=tk.LEFT)
        self.ref_audio_var = tk.StringVar()
        self.ref_audio_entry = ttk.Entry(audio_row, textvariable=self.ref_audio_var)
        self.ref_audio_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        ttk.Button(audio_row, text="浏览...", command=self._on_browse_audio).pack(side=tk.LEFT, padx=(0, 4))
        self.record_btn = ttk.Button(audio_row, text="开始录音", command=self._on_record)
        self.record_btn.pack(side=tk.LEFT)

        # Reference text
        ttk.Label(ref_frame, text="要克隆的音频中对应的文本：").pack(anchor=tk.W)
        self.ref_text = tk.Text(ref_frame, height=4, wrap=tk.WORD)
        self.ref_text.pack(fill=tk.X, pady=(4, 0))

        # Synthesis input frame
        synth_frame = ttk.LabelFrame(container, text="要合成的文本：", padding=12)
        synth_frame.pack(fill=tk.BOTH, expand=True, pady=8)
        self.input_text = tk.Text(synth_frame, height=6, wrap=tk.WORD)
        self.input_text.pack(fill=tk.BOTH, expand=True)

        # Actions row
        actions = ttk.Frame(container)
        actions.pack(fill=tk.X, pady=8)
        self.synthesize_btn = ttk.Button(actions, text="开始合成", command=self._on_synthesize)
        self.synthesize_btn.pack(side=tk.LEFT)

        self.progress = ttk.Progressbar(actions, mode="indeterminate")
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=12)

        self.play_btn = ttk.Button(actions, text="播放音频", command=self._on_play, state=tk.DISABLED)
        self.play_btn.pack(side=tk.RIGHT)

        # Output info
        output_frame = ttk.Frame(container)
        output_frame.pack(fill=tk.X)
        ttk.Label(output_frame, text="输出文件：").pack(side=tk.LEFT)
        self.output_var = tk.StringVar(value="正在加载模型，请耐心等待...")
        self.output_label = ttk.Label(output_frame, textvariable=self.output_var)
        self.output_label.pack(side=tk.LEFT, padx=(4, 0))

        # 启动后台加载模型
        threading.Thread(target=self._load_model_background, daemon=True).start()

    def _center_window(self, width: int, height: int) -> None:
        self.root.update_idletasks()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = int((screen_w - width) / 2)
        y = int((screen_h - height) / 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _on_browse_audio(self) -> None:
        path = filedialog.askopenfilename(
            title="选择参考音频 (WAV)",
            filetypes=[("WAV 文件", "*.wav"), ("所有文件", "*.*")],
        )
        if path:
            self.reference_audio_path = path
            self.ref_audio_var.set(path)

    def _validate_inputs(self) -> tuple[bool, str]:
        audio_path = (self.reference_audio_path or "").strip()
        if not audio_path or not os.path.isfile(audio_path):
            return False, "请先选择有效的参考音频文件。"
        ref_text = self.ref_text.get("1.0", tk.END).strip()
        if not ref_text:
            return False, "请填写参考音频中对应的文本。"
        input_text = self.input_text.get("1.0", tk.END).strip()
        if not input_text:
            return False, "请输入要合成的问题或文本。"
        return True, ""

    def _load_model_background(self) -> None:
        try:
            # 后台加载模型
            with self.model_lock:
                if self.tts_model is None:
                    logger.info("正在加载模型...")
                    self.tts_model = NeuTTSAir(
                        backbone_repo=self.model_dir,
                        backbone_device=self.device,
                        codec_repo="./models/neucodec",
                        codec_device=self.device,
                        language=self.language,
                    )
                    logger.info("模型加载完成")

            def on_loaded() -> None:
                self.model_loaded = True
                messagebox.showinfo("模型就绪", "模型已加载完成，可以开始合成。")

            self.root.after(0, on_loaded)
        except Exception as e:
            def on_error() -> None:
                self.model_loaded = False
                messagebox.showerror("模型加载失败", f"请检查模型路径与依赖。\n错误：{e}")

            self.root.after(0, on_error)

    def _on_synthesize(self) -> None:
        if not self.model_loaded:
            messagebox.showwarning("提示", "模型尚未加载完成，请稍候。")
            return
        ok, msg = self._validate_inputs()
        if not ok:
            messagebox.showwarning("提示", msg)
            return

        # Lock UI
        self.synthesize_btn.config(state=tk.DISABLED)
        self.play_btn.config(state=tk.DISABLED)
        self.progress.start(12)

        reference_text = self.ref_text.get("1.0", tk.END).strip()
        input_text = self.input_text.get("1.0", tk.END).strip()
        reference_audio_path = self.reference_audio_path or ""

        def worker() -> None:
            try:
                start_time = time.time()
                # 编码参考
                ref_codes = self.tts_model.encode_reference(reference_audio_path)
                # 合成
                wav = self.tts_model.infer(input_text, ref_codes, reference_text)

                # 保存
                os.makedirs("output", exist_ok=True)
                timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                out_path = os.path.join("output", f"tts-{timestamp}.wav")
                sf.write(out_path, wav, 24000)

                def on_done() -> None:
                    self.last_output_path = out_path.replace("\\", "/")
                    self.output_var.set(f"语音已合成，保存在：{self.last_output_path}")
                    self.progress.stop()
                    self.synthesize_btn.config(state=tk.NORMAL)
                    self.play_btn.config(state=tk.NORMAL)
                    logger.info(f"音频已保存：{self.last_output_path}，耗时：{int((time.time() - start_time) * 1000)}ms")

                self.root.after(0, on_done)
            except Exception as e:
                def on_error() -> None:
                    self.progress.stop()
                    self.synthesize_btn.config(state=tk.NORMAL)
                    self.play_btn.config(state=(tk.NORMAL if self.last_output_path else tk.DISABLED))
                    messagebox.showerror("合成失败", f"发生错误：{e}")

                self.root.after(0, on_error)

        threading.Thread(target=worker, daemon=True).start()

    def _on_record(self) -> None:
        """录音按钮点击事件：开始录音或停止录音"""
        if not self.is_recording:
            # 开始录音
            self._start_recording()
        else:
            # 停止录音
            self._stop_recording()
    
    def _start_recording(self) -> None:
        """开始录音"""
        try:
            self.is_recording = True
            self.frames = []
            self.record_btn.config(text="停止录音", state=tk.NORMAL)
            self.synthesize_btn.config(state=tk.DISABLED)
            
            # 在后台线程中录音
            def record_thread() -> None:
                try:
                    # 录音参数
                    chunk = 1024
                    sample_format = pyaudio.paInt16
                    channels = 1
                    sample_rate = 24000  # 与模型使用的采样率一致
                    
                    p = pyaudio.PyAudio()
                    sample_size = p.get_sample_size(sample_format)
                    self.audio_stream = p.open(
                        format=sample_format,
                        channels=channels,
                        rate=sample_rate,
                        frames_per_buffer=chunk,
                        input=True
                    )
                    
                    logger.info("开始录音...")
                    while self.is_recording:
                        data = self.audio_stream.read(chunk)
                        self.frames.append(data)
                    
                    # 停止录音
                    self.audio_stream.stop_stream()
                    self.audio_stream.close()
                    p.terminate()
                    
                    # 保存录音文件
                    os.makedirs("output", exist_ok=True)
                    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                    record_path = os.path.join("output", f"record-{timestamp}.wav")
                    
                    wf = wave.open(record_path, 'wb')
                    wf.setnchannels(channels)
                    wf.setsampwidth(sample_size)
                    wf.setframerate(sample_rate)
                    wf.writeframes(b''.join(self.frames))
                    wf.close()
                    
                    def on_record_done() -> None:
                        self.reference_audio_path = record_path.replace("\\", "/")
                        self.ref_audio_var.set(self.reference_audio_path)
                        self.record_btn.config(text="开始录音", state=tk.NORMAL)
                        self.synthesize_btn.config(state=tk.NORMAL)
                        messagebox.showinfo("录音完成", f"录音已保存：{self.reference_audio_path}")
                        logger.info(f"录音已保存：{self.reference_audio_path}")
                    
                    self.root.after(0, on_record_done)
                except Exception as e:
                    def on_error() -> None:
                        self.is_recording = False
                        self.record_btn.config(text="开始录音", state=tk.NORMAL)
                        self.synthesize_btn.config(state=tk.NORMAL)
                        messagebox.showerror("录音失败", f"录音过程中发生错误：{e}")
                    
                    self.root.after(0, on_error)
                    logger.error(f"录音失败：{e}")
            
            self.recording_thread = threading.Thread(target=record_thread, daemon=True)
            self.recording_thread.start()
        except Exception as e:
            self.is_recording = False
            self.record_btn.config(text="开始录音", state=tk.NORMAL)
            self.synthesize_btn.config(state=tk.NORMAL)
            messagebox.showerror("录音失败", f"无法启动录音：{e}")
            logger.error(f"无法启动录音：{e}")
    
    def _stop_recording(self) -> None:
        """停止录音"""
        self.is_recording = False
        self.record_btn.config(text="正在停止...", state=tk.DISABLED)
    
    def _on_play(self) -> None:
        if not self.last_output_path or not os.path.isfile(self.last_output_path):
            messagebox.showwarning("提示", "暂无可播放的文件，请先完成一次合成。")
            return
        try:
            winsound.PlaySound(self.last_output_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as e:
            messagebox.showerror("播放失败", f"无法播放音频：{e}")


def main() -> None:
    root = tk.Tk()
    app = TTSApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
