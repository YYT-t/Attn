import os
from transformers import AutoTokenizer, Trainer, TrainingArguments
from collators import My_collator
from configs import MyGPT2Config
import torch
import numpy as np

import argparse

from networks import MyGPT2LMHeadModel
from data_structure_related.data_structure import Goal_graph
from utils import *
import logging
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gpt2")
parser.add_argument("--num_icl_train_traces", type=int, default=2000)
parser.add_argument("--num_icl_valid_traces", type=int, default=100)
parser.add_argument("--num_mk_train_traces", type=int, default=2000)
parser.add_argument("--num_mk_valid_traces", type=int, default=100)
parser.add_argument("--max_examples", type=int, default=5)

parser.add_argument("--graph_type", type=int)
parser.add_argument("--graph_len", type=int)
parser.add_argument("--graph_width", type=int)
parser.add_argument("--merge_pos", type=int)

parser.add_argument("--max_child_chain_len", type=int, default=2)
parser.add_argument("--vocab_size", type=int, default=52)
parser.add_argument("--env_val_num_low", type=int, default=10)
parser.add_argument("--chain_val_num", type=int, default=50)
parser.add_argument("--leak_prob_node", type=float, default=0.005)
parser.add_argument("--leak_prob_val", type=float, default=0.005)
parser.add_argument("--tl_low", type=int, default=10)
parser.add_argument("--addlen", type=int, default=2)
parser.add_argument("--nearlen", type=int, default=2)
parser.add_argument("--context_lower", type=int, default=1)
parser.add_argument("--context_upper", type=int, default=7)
parser.add_argument("--context_div", type=int, default=7)


parser.add_argument("--n_layers", type=int, default=1)
parser.add_argument("--n_heads", type=int, default=1) 
parser.add_argument("--hidden_size", type=int, default=100)

parser.add_argument("--if_train", type=str, default="y")
parser.add_argument("--if_upload", type=str, default="y")
parser.add_argument("--train_epoch", type=int, default=10)
parser.add_argument("--per_device_train_batch_size", type=int, default=8)
parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
parser.add_argument("--save_steps", type=int, default=1000)
parser.add_argument("--if_test", type=str, default="y")
parser.add_argument("--if_plot", type=str, default="y")
parser.add_argument("--if_probe", type=str, default="y")
parser.add_argument("--probe_mean_num", type=int, default=10)
parser.add_argument("--test_epoch", type=int)

parser.add_argument("--if_in_colab", type=str, default="n")

Args = parser.parse_args()

print("Args:", Args)

assert Args.graph_len > Args.max_child_chain_len

if Args.if_in_colab=="y":
        os.chdir("/content/drive/MyDrive/ICL/CoT_handin")
        print("Current working directory:", os.getcwd())
Graph_shape = [Args.graph_width]*Args.merge_pos + [1] + [Args.graph_width]*(Args.graph_len-Args.merge_pos-1)
Test_len = len(Graph_shape)
Test_max_examples = Args.max_examples
Tokenizer = AutoTokenizer.from_pretrained(Args.model)
Tokenizer.pad_token = Tokenizer.eos_token
Token_num = len(Tokenizer)
print("Token_num:", Token_num)

My_Goal_graph = Goal_graph(graph_shape=Graph_shape,
                           graph_type=Args.graph_type,
                        context_lower=Args.context_lower,
                        context_upper=Args.context_upper,
                        context_div=Args.context_div,
                           vocab_size=Args.vocab_size,
                          env_val_num_low=Args.env_val_num_low,
                          chain_val_num=Args.chain_val_num,
                                leak_prob_node=Args.leak_prob_node,
                                leak_prob_val=Args.leak_prob_val,
                           addlen=Args.addlen,
                           nearlen=Args.nearlen,
                           tl_low=Args.tl_low,
                           tokenizer=Tokenizer
                           )
data_dir = f"data_and_models"
shape_dir = f"len{Args.graph_len}_width{Args.graph_width}_merge{Args.merge_pos}"
if not os.path.exists(f"{data_dir}/{shape_dir}"):
        os.makedirs(f"{data_dir}/{shape_dir}")
foot_str = f"{shape_dir}/maxchildlen{str(Args.max_child_chain_len)}\
_cl{Args.context_lower}_cu{Args.context_upper}_cd{Args.context_div}_vocab{str(Args.vocab_size)}_envaln{str(Args.env_val_num_low)}\
_chainvaln{str(Args.chain_val_num)}_lkpn{Args.leak_prob_node}_lkpv{Args.leak_prob_val}\
_addlen{Args.addlen}_nearlen{Args.nearlen}_tl{Args.tl_low}_shot{Args.max_examples}_icl{str(Args.num_icl_train_traces)}_mk{str(Args.num_mk_train_traces)}"

type_dir = f"{data_dir}/{foot_str}/type{str(Args.graph_type)}"
if not os.path.exists(type_dir):
        os.makedirs(type_dir)
model_dir = f"{type_dir}/outs_{Args.model}"

Train_ds, Valid_ds = prepare_training_data(My_Goal_graph, Args, Tokenizer, data_dir, type_dir)
Context_len = 2048  #len(Train_ds["input_ids"][0])

Config = MyGPT2Config(
    vocab_size=len(Tokenizer),
    n_ctx=Context_len,
    bos_token_id=Tokenizer.bos_token_id,
    eos_token_id=Tokenizer.eos_token_id,
    n_layer=Args.n_layers,
    n_head=Args.n_heads,
    max_position_embeddings=Context_len,
        hidden_size=Args.hidden_size,
)

Device = "cuda" if torch.cuda.is_available() else "cpu"
Model = MyGPT2LMHeadModel(Config).to(Device)
print("Model:", Model._get_name())
print("Device:", Device)
#print model's parameters number
num_params = sum(p.numel() for p in Model.parameters())
print("num_params:", num_params)
Data_collator = My_collator(Tokenizer)


print("model_dir:", model_dir)
if not os.path.exists(model_dir):
        print("Create model_dir:", model_dir)
        os.makedirs(model_dir)

outs_path = f"{model_dir}/layer{Args.n_layers}_head{Args.n_heads}_hidden{Args.hidden_size}"

print("outs_path:", outs_path)
Train_Args = TrainingArguments(
    output_dir=outs_path,
    eval_strategy="epoch",
    num_train_epochs=Args.train_epoch,
    save_steps=Args.save_steps,
        per_device_eval_batch_size=Args.per_device_eval_batch_size,                                                                                                               
        per_device_train_batch_size=Args.per_device_train_batch_size,   
)

if Args.if_train=="y":
        print("training")
        trainer = Trainer(
        model=Model,
        tokenizer=Tokenizer,
        args=Train_Args,
        data_collator=Data_collator,
        train_dataset=Train_ds,
        eval_dataset=Valid_ds
        )
        trainer.train()

if Args.if_upload=="y":
        def get_latest_checkpoint(outs_path):
                checkpoints = [d for d in os.listdir(outs_path) if d.startswith("checkpoint-")]
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
                return os.path.join(outs_path, latest_checkpoint), latest_checkpoint

        latest_checkpoint, ckpt = get_latest_checkpoint(outs_path)
        # Model = MyGPT2LMHeadModel.from_pretrained(latest_checkpoint, config=Config).to(Device)
        # Model.push_to_hub("YYT-t/layer1_head1_len2_child2_ckpt1")
        print("Latest checkpoint:", latest_checkpoint)
        import subprocess
        subprocess.run([
            "huggingface-cli", "upload", 
             f"YYT-t/layer{Args.n_layers}_head{Args.n_heads}_len{Args.graph_len}_child{Args.max_child_chain_len}_{ckpt}",
             latest_checkpoint,
            "--token", "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"
        ])

if Args.if_test=="y" or Args.if_probe=="y" or Args.if_plot=="y":
        model_path = f"{outs_path}/checkpoint-{Args.test_epoch}"
        Model = MyGPT2LMHeadModel.from_pretrained(model_path, config=Config).to(Device)

if Args.if_test=="y":
        trainer = Trainer(
                model=Model,
                tokenizer=Tokenizer,
                args=Train_Args,
                data_collator=Data_collator,
                train_dataset=Train_ds,
                eval_dataset=Valid_ds
        )
        log_path = f"{outs_path}/test_epoch{Args.test_epoch}_len{Test_len}.log"
        handler = logging.FileHandler(log_path, mode='w')
        handler.setFormatter(logging.Formatter('%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s'))
        logger = logging.getLogger('testing')
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.info(trainer.evaluate())
        test_map = do_test(My_Goal_graph, Model, Tokenizer, Test_max_examples, Test_len, logger)
        tr_ver = np.mean(test_map["tr_ver"][1:])
        tr_val = np.mean(test_map["tr_val"])
        test_ver_0 = test_map["te_ver"][0]
        test_ver_f = max(test_map["te_ver"][1:])
        test_val_0 = test_map["te_val"][0]
        test_val_f = max(test_map["te_val"][1:])
        test_final_0 = test_map["final"][0]
        test_final_f = max(test_map["final"][1:])
        test_whole_0 = test_map["whole"][0]
        test_whole_f = max(test_map["whole"][1:])
        logger.info(f"train vertices acc:{tr_ver}, train value acc:{tr_val}, \
test vertices 0:{test_ver_0}, test vertices f:{test_ver_f}, \
test value 0:{test_val_0}, test value f:{test_val_f}, \
test_final_0:{test_final_0}, test_final_f:{test_final_f}, \
test_whole_0:{test_whole_0}, test_whole_f:{test_whole_f}")

        
if Args.if_probe =="y":
        log_path = f"{outs_path}/prob_epoch{Args.test_epoch}_meannum{Args.probe_mean_num}.log"
        handler = logging.FileHandler(log_path, mode='w')
        handler.setFormatter(logging.Formatter('%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s'))
        logger = logging.getLogger('probing')
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.info(f"path len={Test_len}")

        logger.info("mk probing")
        do_probe(My_Goal_graph, Model, Tokenizer, 5, Args.max_child_chain_len, Test_len, 10, logger, Device, "mk", "val")
        
        logger.info("test probing")
        do_probe(My_Goal_graph, Model, Tokenizer, 5, Args.max_child_chain_len, Test_len, 10, logger, Device, "test", "val")

if Args.if_plot=="y":
        do_plot(Args, My_Goal_graph, Model, Tokenizer, 2, Test_len,  Device, Train_ds, outs_path, Args.test_epoch)