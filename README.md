# The sources codes of DR-BERT and baselines

## Recent Activity
1. Our released RRS corpus can be found [here](https://drive.google.com/drive/folders/1EcjrkDnx8mSZlGum0dQHYJGDyUMEEwBz?usp=sharing).
2. Our released BERT-FP post-training checkpoint for the RRS corpus can be found [here](https://drive.google.com/drive/folders/1PSdIC6H1SCHWhaBxtRjX0029No8kZtBI?usp=sharing).
3. Our related work (Exploring Dense Retrieval for Dialogue Response Selection) can be found [here](https://arxiv.org/pdf/2110.06612.pdf).
4. Our post-training and trained checkpoints are released [here](https://drive.google.com/drive/folders/1y48ky8twFKbvcu9TCJ9DxBG9lMTYnRYI?usp=sharing). Feel free to reproduce the experimental results in the paper.

## How to Use

1. Init the repo

    Before using the repo, please run the following command to init:
    
    ```bash
    # create the necessay folders
    python init.py
    
    # prepare the environment
    pip install -r requirements.txt
    ```

2. train the model

    The necessary details can be found under the `config` folder.

    ```bash
    # dataset_name: douban, ecommerce, ubuntu, restoration-200k
    # model_name: dual-bert(DR-BERT), bert-ft, sa-bert, bert-fp(post-training), poly-encoder
    ./scripts/train.sh <dataset_name> <model_name> <cuda_ids>
    ```

3. test the model

    ```bash
    ./scripts/test_rerank.sh <dataset_name> <model_name> <cuda_id>
    ```
