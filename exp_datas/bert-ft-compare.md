# BERT-FT-Compare Model for Dialog Response Selection and Automatic Evaluation Metric

## 1. Dialog Response Selection

### 1.1 E-Commerce Dataset

| Models             | R10@1 | R10@2 | R10@5 | MRR   |
| ------------------ | ----- | ----- | ----- | ----- |
| SA-BERT+HCL        | 72.1  | 89.6  | 99.3  | -     |
| BERT-FP            | 87.0  | 95.6  | 99.3  | 92.52 |
| bert-ft-compare(margin=0.1)    |  89.2     |   95.4    |    98.6   |    93.49    |
| bert-ft-compare(margin=0.2)    |  90.3     |   95.5    |    98.8   |    94.06    |

### 1.2 Douban Dataset

| Models             | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ |
| SA-BERT+HCL        | 33.0  | 53.1  | 85.8  | 68.1  | 51.4  | 63.9   |
| BERT-FP            | 32.4  | 54.2  | 87.0  | 68.0  | 51.2  | 64.4   |
| bert-ft-compare(margin=0.1)    | 31.0  | 53.37 | 86.76 | 67.45 | 49.63 | 63.24  |
| bert-ft-compare(margin=0.2)    | 31.7  | 53.92 | 85.45 | 68.08 | 50.67 | 63.68  |
| bert-ft-compare(margin=0.3)    | 31.86 | 52.98 | 85.66 | 67.89 | 50.67 | 63.7  |

### 1.3 Ubuntu-v1 Dataset

| Models         | R10@1 | R10@2 | R10@5 | R2@1   |
| -------------- | ----- | ----- | ----- | ------ |
| SA-BERT+HCL    | 86.7  | 94.0  | 99.2  | 97.7   |
| BERT-FP        | 91.1  | 96.2  | 99.4  | 97.7   |
| bert-ft-compare(margin=0.2) |       |       |       |        |

### 1.4 HORSe Dataset

| Models             | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ |
| BERT               | 41.35 | 61.84 | 88.21 | 64.35 | 45.96 | 63.18  |
| SA-BERT            | 44.44 | 65.3  | 92.17 | 66.97 | 48.79 | 66.03  |
| SA-BERT+BERT-FP    | 51.46 | 69.43 | 92.82 | 71.99 | 57.07 | 70.72  |
| BERT-FP            | 49.32 | 69.89 | 91.86 | 70.81 | 54.55 | 69.8   |
| DR-BERT(full)      | 56.44 | 74.7  | 93.56 | 75.91 | 62.42 | 74.75  |
| bert-ft-compare(margin=0.1)  | 89.74 | 96.6   | 98.59 | 97.98 | 97.27 | 97.89  |
| bert-ft-compare(margin=0.1,bi-edge)  | 87.82 | 97.96   | 99.95 | 97.53 | 95.25 | 97.45  |
| bert-ft-compare(margin=0.2)  | 68.93 | 81.88  | 96.84 | 84.32 | 76.06 | 83.15  |


* HORSe ranking test set

| Models           | NDCG@3 | NDCG@5 |
| ---------------- | ------ | ------ |
| BERT             | 0.625  | 0.714  |
| BERT-FP          | 0.609  | 0.709  |
| SA-BERT(BERT-FP) | 0.674  | 0.753  |
| Poly-encoder     | 0.679  | 0.765  |
| DR-BERT(full)    | 0.699  | 0.774  |
| bert-ft-compare  | 0.536  | 0.619  |


## 2. Automatic Evaluation Metric
