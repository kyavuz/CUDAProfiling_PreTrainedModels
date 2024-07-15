# this code uses bert, roberta and gpt pretrained models. CUDA devices used to run.

from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model, RobertaTokenizer, RobertaModel
import torch
import time                                                             # for measuring the execution time
import io                                                               # for opening dataset files
import os                                                               # for dataset file paths

import torchvision.models as models                                     # for PyTorch Cuda Profiler
import torch.nn as nn                                                   # for PyTorch Cuda Profiler
import torch.optim as optim                                             # for PyTorch Cuda Profiler
from torch.profiler import profile, record_function, ProfilerActivity   # for PyTorch Cuda Profiler

def bert(text, device):
    # load Bert Tokenizer and BertModel
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model = model.to(device)

    tokenized_sentence = tokenizer.encode(text, padding=True, truncation=True, max_length=50, add_special_tokens=True)
    # creating input_ids tensor and moves to GPU
    input_ids = torch.tensor([tokenized_sentence]).to(device)
    with torch.no_grad():
        outputs = model(input_ids)

def gpt(text, device):
    # load GPT-2 tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2').to(device)
    
    # To avoid 'ValueError: Asking to pad but the tokenizer does not have a padding token.' error, add pad_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    
    tokenized_sentence = tokenizer.encode(text, padding=True, truncation=True, max_length=50, add_special_tokens=True)
    # creating input_ids tensor and moves to GPU
    input_ids = torch.tensor(tokenized_sentence).to(device)
    with torch.no_grad():
        outputs = model(input_ids)

def roberta(text, device):
    # load RoBERTa tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base').to(device)

    tokenized_sentence = tokenizer.encode(text, padding=True, truncation=True, max_length=50, add_special_tokens=True)
    # creating input_ids tensor and moves to GPU
    input_ids = torch.tensor([tokenized_sentence]).to(device)
    with torch.no_grad():
        outputs = model(input_ids)

    

def main():
    # delete cuda profile output txt files if any exists
    if os.path.exists(os.path.join('pytorch_profiler_results', "bert_profile_results.txt")):
        os.remove(os.path.join('pytorch_profiler_results', "bert_profile_results.txt"))
    if os.path.exists(os.path.join('pytorch_profiler_results', "gpt_profile_results.txt")):
        os.remove(os.path.join('pytorch_profiler_results', "gpt_profile_results.txt"))
    if os.path.exists(os.path.join('pytorch_profiler_results', "roberta_profile_results.txt")):
        os.remove(os.path.join('pytorch_profiler_results', "roberta_profile_results.txt"))
    
    files = ['onlinefoods.csv', 'fuel.csv', 'netflix_shows.csv']

    # check if any cuda device available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*40)
    print("Using device:", device)
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
    
    for i in range(len(files)):
        file_path = os.path.join('datasets', files[i])
        with io.open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        print("="*40)
        print(files[i])
        #print(text)

        # BERT
        start_time = time.time()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
            with record_function("model_inference"):
                bert(text, device)
        end_time = time.time()
        execution_time = end_time - start_time
        print("bert execution time for ", files[i], ":",execution_time)

        # print profile results
        #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
        # for detailed info
        #for evt in prof.key_averages():
        #    print(f"{evt.key}: CPU time: {evt.cpu_time_total:.3f} ms, CUDA time: {evt.cuda_time_total:.3f} ms")

        # write profile results to output file
        with open(os.path.join('pytorch_profiler_results', "bert_profile_results.txt"), "a") as f:
            f.write(f"BERT RESULTS FOR {files[i]}\n")
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            for evt in prof.key_averages():
                f.write(f"{evt.key}: CPU time: {evt.cpu_time_total:.3f} ms, CUDA time: {evt.cuda_time_total:.3f} ms\n")
            f.write("\n")


        # GPT
        start_time = time.time()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
            with record_function("model_inference"):
                gpt(text, device)
        end_time = time.time()
        execution_time = end_time - start_time
        print("gpt execution time for ", files[i], ":",execution_time)

        # print profile results
        #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
        # for detailed info
        #for evt in prof.key_averages():
        #    print(f"{evt.key}: CPU time: {evt.cpu_time_total:.3f} ms, CUDA time: {evt.cuda_time_total:.3f} ms")

        # write profile results to output file
        with open(os.path.join('pytorch_profiler_results', "gpt_profile_results.txt"), "a") as f:
            f.write(f"GPT RESULTS FOR {files[i]}\n")
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            for evt in prof.key_averages():
                f.write(f"{evt.key}: CPU time: {evt.cpu_time_total:.3f} ms, CUDA time: {evt.cuda_time_total:.3f} ms\n")
            f.write("\n")


        # ROBERTA
        start_time = time.time()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
            with record_function("model_inference"):
                roberta(text, device)
        end_time = time.time()
        execution_time = end_time - start_time
        print("roberta execution time for ", files[i], ":",execution_time)

        # print profile results
        #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
        # for detailed info
        #for evt in prof.key_averages():
        #    print(f"{evt.key}: CPU time: {evt.cpu_time_total:.3f} ms, CUDA time: {evt.cuda_time_total:.3f} ms")

        # write profile results to output file
        with open(os.path.join('pytorch_profiler_results', "roberta_profile_results.txt"), "a") as f:
            f.write(f"ROBERTA RESULTS FOR {files[i]}\n")
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            for evt in prof.key_averages():
                f.write(f"{evt.key}: CPU time: {evt.cpu_time_total:.3f} ms, CUDA time: {evt.cuda_time_total:.3f} ms\n")
            f.write("\n")


if __name__ == "__main__":
    main()





