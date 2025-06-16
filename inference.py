import torch
from pathlib import Path
import argparse
from tqdm import tqdm
import os
import json
from collections import OrderedDict

# 确保能导入所有需要的模块
from config import *
from utils import *
from llama import Transformer, ModelArgs
from transformers import LlamaTokenizer, default_data_collator
from accelerate import Accelerator
import datasets


def run_inference(args):
    accelerator = Accelerator()
    # --- 数据加载部分 ---
    print("--- Loading test data and tokenizer ---")
    tokenizer = LlamaTokenizer.from_pretrained('Llama-2-7b-hf')
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'left'
    dataset, split, edge_index = load_dataset[args.dataset]()
    original_dataset = dataset.map(
        preprocess_original_dataset[args.dataset](tokenizer=tokenizer, max_length=original_len[args.dataset]),
        batched=True,
        remove_columns=[i for i in dataset.column_names if i not in ['node_ids']],
    ).with_format("torch")
    clm_dataset_test = dataset.map(
        preprocess_test_dataset[args.dataset](tokenizer=tokenizer, max_length=instruction_len[args.dataset]),
        batched=True,
        remove_columns=[i for i in dataset.column_names if i not in ['node_ids', 'label', 'text_label']],
    ).with_format("torch")
    test_dataset = clm_dataset_test.select(split['test'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.eval_batch_size,
                                              collate_fn=default_data_collator)
    print("Test data loaded.")

    # --- 模型创建与权重加载 ---
    print(f"--- Loading sharded model from '{args.model_path}' ---")

    with open(Path(f"{module_path}/{args.model_name}/") / "params.json", "r") as f:
        params = json.loads(f.read())
    model_args = ModelArgs(w_adapter=True, adapter_dim=args.adapter_dim, adapter_len=args.adapter_len, **params)
    model_args.vocab_size = tokenizer.vocab_size

    # 在CPU上创建完整的模型实例
    inference_model = Transformer(params=model_args, edge_index=edge_index,
                                  input_ids=original_dataset['input_ids'],
                                  input_attention_mask=original_dataset['attention_mask'])

    # 手动加载所有分片文件并合并成一个 state_dict
    state_dict = {}
    index_path = os.path.join(args.model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Cannot find 'model.safetensors.index.json' in {args.model_path}")

    with open(index_path, "r") as f:
        index_data = json.load(f)

    # 拿到权重到文件的映射关系
    weight_map = index_data['weight_map']

    # 建立文件到权重的反向映射
    file_to_weights = {}
    for key, filename in weight_map.items():
        if filename not in file_to_weights:
            file_to_weights[filename] = []
        file_to_weights[filename].append(key)

    # 逐一加载每个分片文件
    from safetensors.torch import load_file
    for filename, keys in tqdm(file_to_weights.items(), desc="Loading shards"):
        shard_path = os.path.join(args.model_path, filename)
        shard_state_dict = load_file(shard_path, device="cpu")
        state_dict.update(shard_state_dict)

    # 处理 'module.' 前缀问题
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = key[7:]  # 移除 'module.' 前缀
        else:
            new_key = key
        new_state_dict[new_key] = value

    # 将修正后的权重加载到模型中
    # 我们需要忽略 input_ids 和 input_attention_mask，因为它们不是模型参数
    inference_model.load_state_dict(new_state_dict, strict=False)
    print("Sharded model manually loaded and keys fixed successfully.")

    # 将完整模型分发到多张卡上进行ZeRO-3推理
    print("Preparing model for ZeRO-3 inference...")
    inference_model, test_loader = accelerator.prepare(
        inference_model, test_loader
    )
    accelerator.print("Model and dataloader prepared.")
    # 为了让每个进程的输出都清晰可见，我们不使用 accelerator.print，而是用普通的 print
    # 这样可以看到所有进程（Rank 0, 1, 2）的输出对比
    # print("\n" + "=" * 60)
    # print(f"--- [Process {accelerator.process_index} | GPU {accelerator.device}] 数据集长度检查 ---")
    #
    # # 1. 打印 `prepare` 之前的、完整的 `test_dataset` 的长度
    # #    这个值在所有进程上都应该是相同的（我们的例子中是 10）
    # print(f"  [Process {accelerator.process_index}] 原始 'test_dataset' 的样本数: {len(test_dataset)}")
    #
    # # 2. 打印 `prepare` 之后的、当前进程的 `test_loader` 包含的批次数
    # #    这个值可能因进程而异
    # print(f"  [Process {accelerator.process_index}] 分发后 'test_loader' 的批次数: {len(test_loader)}")
    #
    # # 3. 打印 `test_loader` 内部的 .dataset 的长度
    # #    这个值最能说明问题，它显示了当前进程实际持有的样本子集的大小
    # #    在不同的进程上，这个值应该是不同的
    # print(f"  [Process {accelerator.process_index}] 分发后 'test_loader.dataset' 的样本数: {len(test_loader.dataset)}")
    #
    # print("=" * 60 + "\n")
    # accelerator.wait_for_everyone()

    # --- 执行推理 ---
    unwrapped_model = accelerator.unwrap_model(inference_model)
    unwrapped_model.eval()
    eval_output = []
    progress_bar_test = tqdm(test_loader, desc="Final Testing")

    for batch in progress_bar_test:
        # device_map="auto" 会自动处理设备，但输入的batch仍需放到主卡上
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}
        with torch.no_grad():
            kwargs = {"node_ids": batch.get('node_ids'), "input_ids": batch['input_ids'],
                      "attention_mask": batch['attention_mask'], "max_new_tokens": 15}
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            generated_tokens = unwrapped_model.generate(**kwargs)
            gathered_tokens = accelerator.gather(generated_tokens)
            # 推理结果在CPU上收集
            eval_output.append(gathered_tokens.cpu().numpy())

    # 5. 计算指标
    # 后处理和计算指标
    eval_decode_output = []
    for batch_output in eval_output:
        eval_decode_output.extend(tokenizer.batch_decode(batch_output, skip_special_tokens=False))
    eval_pred = [item.split('</s>')[0] for item in eval_decode_output]
    eval_pred = [item.split('\n\n###\n\n ')[-1] for item in eval_pred]
    eval_label = test_loader.dataset['text_label']
    eval_pred = eval_pred[:len(eval_label)]
    pred = [_ == f"{eval_label[i]}" for i, _ in enumerate(eval_pred)]
    acc = sum(pred) / len(pred) if pred else 0.0
    print(f"--- Final Test Accuracy: {acc:.4f} ---")
    print("--- Inference complete. ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 必要参数
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model directory.')

    # 与模型结构相关的参数
    parser.add_argument('--dataset', type=str, default='bgm', help='Dataset name.')
    parser.add_argument('--model_name', type=str, default='LLaMA-7B-2', help='Base model name.')
    parser.add_argument('--adapter_len', type=int, default=5)
    parser.add_argument('--adapter_dim', type=int, default=768)
    parser.add_argument('--adapter_n_heads', type=int, default=6)
    parser.add_argument('--n_decoder_layers', type=int, default=4)
    parser.add_argument('--n_encoder_layers', type=int, default=4)
    parser.add_argument('--n_mp_layers', type=int, default=4)
    parser.add_argument('--rrwp', type=int, default=8)

    # 推理相关的参数
    parser.add_argument('--eval_batch_size', type=int, default=32, help='Batch size for evaluation.')

    inf_args = parser.parse_args()
    run_inference(inf_args)
