# Easy-to-use toolkit for retrieval-based Chatbot

## Recent Activity
1. Our released RRS corpus can be found [here](https://drive.google.com/drive/folders/1EcjrkDnx8mSZlGum0dQHYJGDyUMEEwBz?usp=sharing).
2. Our released BERT-FP post-training checkpoint for the RRS corpus can be found [here](https://drive.google.com/drive/folders/1PSdIC6H1SCHWhaBxtRjX0029No8kZtBI?usp=sharing).
3. Our related work (Exploring Dense Retrieval for Dialogue Response Selection) can be found [here](https://arxiv.org/pdf/2110.06612.pdf).

## How to Use

1. Init the repo

    Before using the repo, please run the following command to init:
    
    ```bash
    # create the necessay folders
    python init.py
    
    # prepare the environment
    # if some package cannot be installed, just google and install it from other ways
    pip install -r requirements.txt
    ```

2. train the model

    ```bash
    ./scripts/train.sh <dataset_name> <model_name> <cuda_ids>
    ```

3. test the model [rerank]

    ```bash
    ./scripts/test_rerank.sh <dataset_name> <model_name> <cuda_id>
    ```

4. test the model [recal]

    ```bash
    # different recall_modes are available: q-q, q-r
    ./scripts/test_recall.sh <dataset_name> <model_name> <cuda_id>
    ```

5. inference the responses and save into the faiss index

    Somethings inference will missing data samples, please use the 1 gpu (faiss-gpu search use 1 gpu quickly)

    It should be noted that:
        1. For writer dataset, use `extract_inference.py` script to generate the inference.txt
        2. For other datasets(douban, ecommerce, ubuntu), just `cp train.txt inference.txt`. The dataloader will automatically read the test.txt to supply the corpus. 

    ```bash
    # work_mode=response, inference the response and save into faiss (for q-r matching) [dual-bert/dual-bert-fusion]
    # work_mode=context, inference the context to do q-q matching
    # work_mode=gray, inference the context; read the faiss(work_mode=response has already been done), search the topk hard negative samples; remember to set the BERTDualInferenceContextDataloader in config/base.yaml
    ./scripts/inference.sh <dataset_name> <model_name> <cuda_ids>
    ```

    If you want to generate the gray dataset for the dataset:

    ```bash
    # 1. set the mode as the **response**, to generate the response faiss index; corresponding dataset name: BERTDualInferenceDataset;
    ./scripts/inference.sh <dataset_name> response <cuda_ids>

    # 2. set the mode as the **gray**, to inference the context in the train.txt and search the top-k candidates as the gray(hard negative) samples; corresponding dataset name: BERTDualInferenceContextDataset
    ./scripts/inference.sh <dataset_name> gray <cuda_ids>

    # 3. set the mode as the **gray-one2many** if you want to generate the extra positive samples for each context in the train set, the needings of this mode is the same as the **gray** work mode
    ./scripts/inference.sh <dataset_name> gray-one2many <cuda_ids>
    ```

    If you want to generate the pesudo positive pairs, run the following commands:

    ```bash
    # make sure the dual-bert inference dataset name is BERTDualInferenceDataset
    ./scripts/inference.sh <dataset_name> unparallel <cuda_ids>
    ```

6. deploy the rerank and recall model

    ```bash
    # load the model on the cuda:0(can be changed in deploy.sh script)
    ./scripts/deploy.sh <cuda_id>
    ```
    at the same time, you can test the deployed model by using:

    ```bash
    # test_mode: recall, rerank, pipeline
    ./scripts/test_api.sh <test_mode> <dataset>
    ```

7. test the recall performance of the elasticsearch

    Before testing the es recall, make sure the es index has been built:
    ```bash
    # recall_mode: q-q/q-r
    ./scripts/build_es_index.sh <dataset_name> <recall_mode>
    ```

    ```bash
    # recall_mode: q-q/q-r
    ./scripts/test_es_recall.sh <dataset_name> <recall_mode> 0
    ```

8. simcse generate the gray responses

    ```bash
    # train the simcse model
    ./script/train.sh <dataset_name> simcse <cuda_ids>
    ```

    ```bash
    # generate the faiss index, dataset name: BERTSimCSEInferenceDataset
    ./script/inference_response.sh <dataset_name> simcse <cuda_ids>
    ```

    ```bash
    # generate the context index
    ./script/inference_simcse_response.sh <dataset_name> simcse <cuda_ids>
    # generate the test set for unlikelyhood-gen dataset
    ./script/inference_simcse_unlikelyhood_response.sh <dataset_name> simcse <cuda_ids>
    ```

    ```bash
    # generate the gray response
    ./script/inference_gray_simcse.sh <dataset_name> simcse <cuda_ids>
    # generate the test set for unlikelyhood-gen dataset
    ./script/inference_gray_simcse_unlikelyhood.sh <dataset_name> simcse <cuda_ids>
    ```
