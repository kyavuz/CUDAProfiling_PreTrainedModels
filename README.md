# CUDAProfiling_PreTrainedModels
#######################################################################################
## Performance Evaluation of Transformer Models in Deep Learning
### Kagan Yavuz
#
### Overview
In recent years, transformer models have revolutionized various natural language processing (NLP) tasks, setting new benchmarks in performance and efficiency. Among these, BERT (Bidirectional Encoder Representations from Transformers), RoBERTa (Robustly optimized BERT approach), and GPT (Generative Pre-trained Transformer) stand out as seminal contributions to the field. Leveraging the attention mechanism and self-attention layers, transformer architectures have enabled deep learning models to capture intricate dependencies and relationships in sequential data, particularly suited for NLP tasks such as text classification, sentiment analysis, and language generation. 

This project is designed to evaluate the performance differences of pre-trained transformer models such as BERT, RoBERTa, and GPT. The models are implemented using PyTorch and evaluated on various datasets to measure execution time and identify performance bottlenecks.

### Methodology
#### Model Selection
For this study, three pre-trained transformer-based models have been selected: BERT (Bidirectional Encoder Representations from Transformers), RoBERTa (Robustly optimized BERT approach), and GPT (Generative Pre-trained Transformer). Each of these models offers unique strengths and has been pre-trained on large corpora to learn rich contextual representations of text data.

#### Implementation
For BERT, GPT, and RoBERTa, the respective tokenizers and models provided by the Hugging Face Transformers library were employed. The tokenizers were used for encoding input text into tokenized sequences, while the models were responsible for generating contextualized representations of the input text. Additionally, the PyTorch library was utilized for tensor operations and GPU acceleration. 

The execution time of each model was measured across various datasets to evaluate their efficiency and scalability across different domains. The evaluation encompassed datasets such as online food reviews (onlinefoods.csv), fuel consumption records (fuel.csv), and Netflix user data (netflix_shows.csv).
* In the project, "datasets" folder contains the datasets used for evaluation. All the datasets are obtained from Kaggle.com.
  * onlinefoods.csv   -> https://www.kaggle.com/code/tayyabli/online-food-dataset#notebook-container
  * fuel.csv          -> https://www.kaggle.com/datasets/sahirmaharajj/fuel-economy
  * netflix_shows.csv -> https://www.kaggle.com/datasets/muhammadkashif724/netflix-tv-shows-2021

For BERT analysis, the input text was tokenized using the BERT tokenizer and then fed into the pre-trained “bert-base-uncased” model to obtain contextualized representations. The execution time for BERT analysis was measured for each dataset. 

Similarly, for GPT analysis, the input text was tokenized using the GPT tokenizer and then processed through the pretrained “gpt2” model to generate contextualized representations. The execution time of GPT-2 analysis was recorded for each dataset. 

Lastly, for RoBERTa analysis, the input text was tokenized using the RoBERTa tokenizer and then the pre-trained “roberta-base” model was employed to generate contextualized representations. The execution time for RoBERTa analysis was measured for each dataset. 

### Initial Performance Measurements
In this section, the selected transformer models—BERT, RoBERTa, and GPT—are evaluated on NVIDIA A100-SXM4-80GB GPU, encompassing execution time metrics across diverse datasets. 

The performance of the BERT, RoBERTa, and GPT models was evaluated in terms of execution time across the three datasets. The execution time of each model was recorded during inference phases using Python time library. The performance measurements presented below, detailing the execution time of each model on the respective datasets. 

onlinefoods.csv
| Model         | Execution Time |
| ------------- | -------------- |
| BERT          | 3,821 sec      |
| GPT           | 2,027 sec      |
| RoBERTa       | 1,751 sec      |

fuel.csv
| Model         | Execution Time |
| ------------- | -------------- |
| BERT          | 22,165 sec      |
| GPT           | 15,595 sec      |
| RoBERTa       | 15,847 sec      |

netflix_shows.csv
| Model         | Execution Time |
| ------------- | -------------- |
| BERT          | 2,655 sec      |
| GPT           | 1,755 sec      |
| RoBERTa       | 1,736 sec      |

### Detailed System Level and Kernel Level Performance Analysis 
In this part, the subsequent analyses were conducted using the NVIDIA GeForce RTX4060 Max-Performance Laptop GPU instead of the Nvidia A100 GPU. 

Based on the PyTorch CUDA Profiler results of the models, the three most time consuming kernels have been detected. Profiler results of the models are given in the "Performance Evaluation of Transformer Models in Deep Learning.pdf" file. 
#

More info about this project can be found in the "Performance Evaluation of Transformer Models in Deep Learning.pdf" file.
#

Thank you,

Kagan Yavuz
