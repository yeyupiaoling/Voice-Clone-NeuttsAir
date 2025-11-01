import re
import warnings
from functools import partial

import torch
from datasets import load_dataset
from loguru import logger
from omegaconf import OmegaConf
from phonemizer.backend import EspeakBackend
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, default_data_collator

warnings.filterwarnings("ignore")


def data_filter(sample):
    text = sample["text"]

    if len(text) == 0:
        return False

    if re.search(r'\d', text):
        return False

    if '£' in text or '$' in text or '¥' in text:
        return False

    return True


def preprocess_sample(sample, tokenizer, max_len, g2p):
    # 获取特殊标记
    speech_gen_start = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_START|>')
    ignore_index = -100  # 这是来自 LLaMA 的

    # 解包样本
    vq_codes = sample["codes"]
    text = sample["text"]

    # 音素化
    phones = g2p.phonemize([text])

    # 安全检查
    if not phones or not phones[0]:
        logger.warning(f"⚠️ 样本的音素化输出为空: {sample['__key__']} text={text}")
        return None

    phones = phones[0].split()
    phones = ' '.join(phones)

    codes_str = "".join([f"<|speech_{i}|>" for i in vq_codes])

    # 获取聊天格式
    chat = f"""user: Convert the text to speech:<|TEXT_PROMPT_START|>{phones}<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}<|SPEECH_GENERATION_END|>"""
    ids = tokenizer.encode(chat)

    # 填充以使其达到序列长度
    if len(ids) < max_len:
        ids = ids + [tokenizer.pad_token_id] * (max_len - len(ids))
    else:
        ids = ids[:max_len]

    # 转换为张量
    input_ids = torch.tensor(ids, dtype=torch.long)

    labels = torch.full_like(input_ids, ignore_index)
    speech_gen_start_idx = (input_ids == speech_gen_start).nonzero(as_tuple=True)[0]
    if len(speech_gen_start_idx) > 0:
        speech_gen_start_idx = speech_gen_start_idx[0]
        labels[speech_gen_start_idx:] = input_ids[speech_gen_start_idx:]

    # 创建注意力掩码
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    # 返回训练所需的格式数据
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def main(config_fpath: str):
    # 加载配置
    logger.info(f"正在从 {config_fpath} 加载配置")
    config = OmegaConf.load(config_fpath)
    checkpoints_dir = config.save_root
    logger.info(f"日志保存到: {checkpoints_dir}")

    restore_from = config.restore_from

    g2p = EspeakBackend(
        language=config.language,
        preserve_punctuation=True,
        with_stress=True,
        words_mismatch="ignore",
        language_switch="remove-flags"
    )
    logger.info(f"正在从 {restore_from} 加载检查点")
    tokenizer = AutoTokenizer.from_pretrained(restore_from)
    model = AutoModelForCausalLM.from_pretrained(restore_from, dtype="auto")

    partial_preprocess = partial(
        preprocess_sample,
        tokenizer=tokenizer,
        max_len=config.max_seq_len,
        g2p=g2p,
    )
    # 加载数据集
    emilia_dataset = load_dataset("json", data_files=config.data_files, split="train")
    emilia_dataset = emilia_dataset.filter(data_filter).map(partial_preprocess, remove_columns=["text", "codes"])
    logger.info(f"数据集加载完成，一共有 {len(emilia_dataset)} 条样本")

    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        do_train=True,
        learning_rate=config.lr,
        num_train_epochs=config.num_train_epochs,
        bf16=True,
        per_device_train_batch_size=config.per_device_train_batch_size,
        warmup_ratio=config.warmup_ratio,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        report_to="none",
        save_strategy="steps",
        ignore_data_skip=True,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        torch_compile=True,
    )

    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=emilia_dataset,
        data_collator=default_data_collator,
    )
    # 开始训练
    trainer.train()
    trainer.save_model(checkpoints_dir)


if __name__ == "__main__":
    main("finetune.yaml")
