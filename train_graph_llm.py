import json
import pathlib
import pickle
import transformers
import torch
import os
import copy
import wandb
import gc
from tqdm import tqdm
from pathlib import Path
from accelerate import Accelerator
from config import *
from transformers import default_data_collator, LlamaTokenizer
from accelerate import DistributedDataParallelKwargs
from accelerate.state import AcceleratorState

from utils import *
from llama import Transformer, ModelArgs
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs
from accelerate import load_checkpoint_and_dispatch

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)

import deepspeed

def main(args, SEED):
    group = f"{args.dataset}"
    accelerator.init_trackers(project_name=f"{args.project}",
                              init_kwargs={"wandb":
                                               {"tags": [args.dataset, args.model_name],
                                                "group": group,
                                                "name": f"{args.dataset}_EXP{SEED}",
                                                "config": args}
                                           },
                              )

    seed_everything(seed=SEED)
    accelerator.print(args)


    with accelerator.main_process_first():
        tokenizer = LlamaTokenizer.from_pretrained('Llama-2-7b-hf')
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = 'left'

        dataset, split, edge_index = load_dataset[args.dataset]()

        original_dataset = dataset.map(
            preprocess_original_dataset[args.dataset](tokenizer=tokenizer, max_length=original_len[args.dataset]),
            batched=True,
            batch_size=None,
            remove_columns=[i for i in dataset.column_names if i not in ['node_ids']],
            keep_in_memory=True,
            writer_batch_size=10000,
            num_proc=os.cpu_count()//2,
        ).with_format("torch")

        clm_dataset_train = dataset.map(
            preprocess_train_dataset[args.dataset](tokenizer=tokenizer, max_length=instruction_len[args.dataset]),
            batched=True,
            batch_size=None,
            remove_columns=[i for i in dataset.column_names if i not in ['node_ids']],
            keep_in_memory=True,
            writer_batch_size=10000,
            num_proc=os.cpu_count()//2,
        ).with_format("torch")


        clm_dataset_test = dataset.map(
            preprocess_test_dataset[args.dataset](tokenizer=tokenizer, max_length=instruction_len[args.dataset]),
            batched=True,
            batch_size=None,
            remove_columns=[i for i in dataset.column_names if i not in ['node_ids', 'label', 'text_label']],
            keep_in_memory=True,
            writer_batch_size=10000,
            num_proc=os.cpu_count()//2,
        ).with_format("torch")




    accelerator.wait_for_everyone()

    # Step 2: Build Node Classification Dataset
    train_dataset = clm_dataset_train.select(split['train'])
    val_dataset = clm_dataset_train.select(split['valid'])
    val_dataset_eval = clm_dataset_test.select(split['valid'])
    test_dataset = clm_dataset_test.select(split['test'])


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True,
                                               pin_memory=True, shuffle=True, collate_fn=default_data_collator)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, drop_last=False, pin_memory=True,
                                             shuffle=False, collate_fn=default_data_collator)
    val_loader_eval = torch.utils.data.DataLoader(val_dataset_eval, batch_size=args.batch_size, drop_last=False,
                                                  pin_memory=True, shuffle=False, collate_fn=default_data_collator)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.eval_batch_size, drop_last=False,
                                              pin_memory=True, shuffle=False, collate_fn=default_data_collator)


    with open(Path(f"{module_path}/{args.model_name}/") / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(w_lora=False,
                                      w_adapter=True,
                                      adapter_layer=8,
                                      adapter_dim=args.adapter_dim,
                                      adapter_len=args.adapter_len,
                                      lora_alpha=16,
                                      lora_r=8,
                                      num_hops=3,
                                      n_mp_layers=args.n_mp_layers,
                                      rrwp=args.rrwp,
                                      n_encoder_layers=args.n_encoder_layers,
                                      n_decoder_layers=args.n_decoder_layers,
                                      adapter_n_heads=args.adapter_n_heads,
                                      task_level=task_level[args.dataset],
                                      **params)


    model_args.vocab_size = tokenizer.vocab_size
    is_zero3 = False
    if accelerator.distributed_type == "DEEPSPEED":
        if AcceleratorState().deepspeed_plugin.zero_stage == 3:
            is_zero3 = True
            accelerator.print("ZeRO-3 is enabled. Using deepspeed.zero.Init() for model instantiation.")

    with deepspeed.zero.Init(enabled=is_zero3):
        # torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        base_model: Transformer = Transformer(params=model_args, edge_index=edge_index,
                                              input_ids=original_dataset['input_ids'],
                                              input_attention_mask=original_dataset['attention_mask'],
                                              )
        # torch.set_default_tensor_type(torch.FloatTensor)

    accelerator.print(model_args)

    # Step 4 Set Optimizer
    param_adapter, param_lora = base_model.set_trainable_params_new()


    lr_group = {
        'adapter': args.lr,
        'lora': args.lr,
    }

    wd_group = {
        'adapter': args.wd,
        'lora': args.wd,
    }

    accelerator.print(lr_group)
    accelerator.print(wd_group)

    optimizer = torch.optim.AdamW(
        [
            {'params': param_adapter, 'lr': lr_group['adapter'], 'weight_decay': wd_group['adapter']},
            *([{'params': param_lora, 'lr': lr_group['lora'], 'weight_decay': wd_group['lora']}]
              if len(param_lora) else [])
        ],
        betas=(0.9, 0.95))
    num_groups = len(optimizer.param_groups)
    if not args.save_dir:
        print("---Warning: No save directory specified. Model will not be saved after training.---")
    model, train_loader, val_loader, val_loader_eval, optimizer = accelerator.prepare(base_model, train_loader,
                                                                                      val_loader, val_loader_eval,
                                                                                      optimizer)
    ckpt_path = Path(f"{module_path}/{args.model_name}/consolidated.00.pth")
    accelerator.print(f"Loading checkpoint from {ckpt_path} AFTER accelerator.prepare()")
    # --- 使用 accelerator.unwrap_model(model) ---
    # 这会返回被 DeepSpeed 和 Accelerate 完全初始化后的原始 Transformer 模型
    unwrapped_model = accelerator.unwrap_model(model)
    trainable_params, all_param = unwrapped_model.print_trainable_params()

    # 添加一个保护性检查，防止意外的 ZeroDivisionError
    if all_param > 0:
        accelerator.print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
    else:
        accelerator.print("Trainable parameter count will be available after the first forward pass in ZeRO-3.")

    # Step 5. Training
    num_training_steps = args.num_epochs * len(train_loader)
    progress_bar = tqdm(range(num_training_steps))
    best_val_loss, best_val_acc = float('inf'), -float('inf')


    for epoch in range(args.num_epochs):

        model.train()
        epoch_loss, accum_loss = 0., 0.

        for step, batch in enumerate(train_loader):

            # 与Zero3不兼容
            #with accelerator.accumulate(model):
            optimizer.zero_grad()
            loss = model(**batch)
            accelerator.backward(loss)

            accelerator.clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
            if num_groups == 2:
                accelerator.clip_grad_norm_(optimizer.param_groups[1]['params'], 0.1)

            if (step + 1) % args.grad_steps == 0:
                adjust_learning_rate(optimizer.param_groups[0], lr_group['adapter'], step / len(train_loader) + epoch,
                                     args)
                if num_groups == 2:
                     adjust_learning_rate(optimizer.param_groups[1], lr_group['lora'], step / len(train_loader) + epoch,
                                     args)

            optimizer.step()
            epoch_loss, accum_loss = epoch_loss + loss.item(), accum_loss + loss.item()


            if (step + 1) % args.grad_steps == 0:
                adapter_lr = optimizer.param_groups[0]["lr"]
                if num_groups == 2:
                    lora_lr = optimizer.param_groups[1]["lr"]
                    accelerator.log({'Adapter Lr': adapter_lr, 'Lora Lr': lora_lr})
                accelerator.log({'Accum Loss': accum_loss / args.grad_steps})
                accelerator.print(f"Accum Loss: {accum_loss / args.grad_steps}")
                accum_loss = 0.

            progress_bar.update(1)

        accelerator.print(
            f"Epoch: {epoch}|{args.num_epochs}: Train Loss (Epoch Mean): {epoch_loss / len(train_loader)}")
        accelerator.log({'Train Loss (Epoch Mean)': epoch_loss / len(train_loader)})


        val_loss = 0.
        samples_seen = 0
        eval_output = []
        model.eval()

        with torch.no_grad():
            # 使用 tqdm 包装 val_loader
            for step, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch} Val Loss", disable=not accelerator.is_local_main_process)):
                loss = model(**batch)
                val_loss += loss.item()

            accelerator.print(f"Epoch: {epoch}|{args.num_epochs}: Val Loss: {val_loss / len(val_loader)}")
            accelerator.log({'Val Loss': val_loss / len(val_loader)})

            # 使用 tqdm 包装 val_loader_eval
            for step, batch in enumerate(tqdm(val_loader_eval, desc=f"Epoch {epoch} Generating", disable=not accelerator.is_local_main_process)):
                kwargs = {}
                kwargs.update(
                    {"node_ids": batch['node_ids'], "input_ids": batch['input_ids'],
                     "attention_mask": batch['attention_mask'], "max_new_tokens": 15})

                generated_tokens = accelerator.unwrap_model(model).generate(**kwargs)
                # 在 gather 之前，先将不同进程的 token 补齐到相同长度
                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                generated_tokens_gathered = accelerator.gather(generated_tokens).cpu().numpy()

                if accelerator.num_processes > 1:
                    if step == len(val_loader_eval) - 1:
                        generated_tokens_gathered = generated_tokens_gathered[
                                                    : len(val_loader_eval.dataset) - samples_seen]
                    else:
                        samples_seen += len(generated_tokens_gathered)
                eval_output.append(generated_tokens_gathered)



        eval_decode_output = []
        for batch_output in eval_output:
            eval_decode_output.extend(tokenizer.batch_decode(batch_output, skip_special_tokens=False))

        eval_pred = [item.split('</s>')[0] for item in eval_decode_output]
        eval_pred = [item.split('\n\n###\n\n ')[-1] for item in eval_pred]

        eval_label = val_loader_eval.dataset['text_label']
        eval_pred = eval_pred[:len(eval_label)]
        pred = [_ == f"{eval_label[i]}" for i, _ in enumerate(eval_pred)]
        val_acc = sum(pred) / len(pred)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            # 只有在指定了保存目录时才执行保存操作
            if args.save_dir:
                # 直接定义最终的保存路径
                if accelerator.is_main_process:
                    print(f"\nEpoch {epoch} New best epoch found! Saving model...")
                output_dir = Path(args.save_dir) / f"{args.dataset}_{args.model_name}_seed{SEED}_best_model"
                # accelerator.save_model 会高效地保存完整模型
                accelerator.save_model(model, output_dir)
                if accelerator.is_main_process:
                    print(
                        f"\nEpoch {epoch}: New best validation accuracy: {val_acc:.4f}. Model saved to '{output_dir}'.")

        accelerator.print(f'Epoch {epoch} Val Acc {val_acc} Best Val Acc {best_val_acc} Best Epoch {best_epoch}')
        accelerator.log({'val acc': val_acc})

    accelerator.wait_for_everyone()
    print(f"Best Val Acc: {best_val_acc} at Epoch {best_epoch}")
    print(f"--- Training completed for seed {SEED} ---")
    if args.save_dir:
        if accelerator.is_main_process:
            print(f"Best model saved at {args.output_dir}.")
    else:
        if accelerator.is_main_process:
            print("Save directory not specified. Model not saved.")


if __name__ == "__main__":
    args = parse_args_llama()
    for exp, SEED in enumerate(range(args.exp_num)):
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        transformers.logging.set_verbosity_error()
        accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs, init_kwargs],
                                  gradient_accumulation_steps=args.grad_steps)
        main(args, SEED)
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        gc.collect()
