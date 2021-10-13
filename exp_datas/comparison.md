# The experiments of the Comparison-based Dialog Response Selection (DRS)

replace the NSP (next sentence prediction) with the CCE (Candidate comparison evaluation) learning object.

## 1. Traditional Settings

In traditional settings, the comparison-based DRS model use the fully comparison to generate the scores for ranking.

### 1.1 E-Commerce Dataset

| Models             | R10@1 | R10@2 | R10@5 | MRR   |
| ------------------ | ----- | ----- | ----- | ----- |
| SA-BERT+HCL        | 72.1  | 89.6  | 99.3  | -     |
| BERT-FP            | 87.0  | 95.6  | 99.3  | 92.52 |
| BERT-CCE           |       |       |       |       |

### 1.2 Douban Dataset

<!-- seed=0; bsz=64; max_len=256,64; epoch=10; lr=5e-5; warmup_ratio=0.0l grad_clip=5.0;-->
| Models             | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ |
| SA-BERT+HCL        | 33.0  | 53.1  | 85.8  | 68.1  | **51.4**  | 63.9   |
| BERT-FP            | 32.4  | 54.2  | **87.0**  | 68.0  | 51.2  | **64.4**   |
| BERT-CCE           | 32.79 | 53.46 | 87.24 | 68.62 | 51.72 | 64.67 |

### 1.3 Ubuntu-v1 Dataset

<!-- seed=0; bsz=64; max_len=256,64; epoch=10(bert-ft=5); lr=5e-5; warmup_ratio=0.0l grad_clip=5.0;-->
| Models         | R10@1 | R10@2 | R10@5 | R2@1   |
| -------------- | ----- | ----- | ----- | ------ |
| SA-BERT+HCL    | 86.7  | 94.0  | 99.2  | 97.7   |
| BERT-FP        | **91.1**  | **96.2**  | **99.4**  | **97.7**   |
| BERT-CCE       |       |       |       |       |

### 1.4 Restoration-200k Dataset (Optional)

* ES test set

<!-- + means the post-train has been used;
bert-fp parameters: lr=3e-5; grad_clip=5.0; see0; batch_size=96; max_len=256, min_mask_num=2;
max_mask_num=20; masked_lm_prob=0.15; min_context_length=2; min_token_length=20; epoch=25; warmup_ratio=0.01-->
| Models             | R10@1 | R10@2 | R10@5 | MRR   |
| ------------------ | ----- | ----- | ----- | ----- |
| dual-bert          | 45.08 | 61.74 | 87.38 | 62.17 |
| bert-ft            | 39.22 | 56.6  | 84.54 | 57.63 |
| BERT-CCE           |       |       |       |       |

* ES test set with human label

| Models             | R10@1 | R10@2 | R10@5 | MRR   |
| ------------------ | ----- | ----- | ----- | ----- |
| dual-bert          | | | | |
| bert-ft            | | | | |
| BERT-CCE           |       |       |       |       |

## 2. Two-stage Boosting for Existing DRS models

### 2.1 E-Commerce Dataset

| Models             | R10@1 | R10@2 | R10@5 | MRR   |
| ------------------ | ----- | ----- | ----- | ----- |
| UMS-BERT           |       |       |       |       |
| UMS-BERT+CCE       |       |       |       |       |
| BERT-FP            |       |       |       |       |
| BERT-FP+CCE        |       |       |       |       |
| dual-bert          |       |       |       |       |
| dual-bert+CCE      |       |       |       |       |

### 2.2 Douban Dataset

<!-- seed=0; bsz=64; max_len=256,64; epoch=10; lr=5e-5; warmup_ratio=0.0l grad_clip=5.0;-->
| Models             | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ |
| BERT-FP            | 32.4  | 54.2  | **87.0**  | 68.0  | 51.2  | **64.4**   |
| BERT-FP+CCE        | 33.0  | 53.56 | **86.65**  | 68.72  | 52.17  | **64.41**   |
| UMS-BERT           |       |       |       |       |       |        |
| UMS-BERT+CCE       |       |       |       |       |       |        |
| dual-bert          |       |       |       |       |       |        |
| dual-bert+CCE      |       |       |       |       |       |        |

### 2.3 Ubuntu-v1 Dataset

<!-- seed=0; bsz=64; max_len=256,64; epoch=10(bert-ft=5); lr=5e-5; warmup_ratio=0.0l grad_clip=5.0;-->
| Models         | R10@1 | R10@2 | R10@5 | R2@1   |
| -------------- | ----- | ----- | ----- | ------ |
| BERT-FP        | **91.1**  | **96.2**  | **99.4**  | **97.7**   |
| BERT-FP+CCE    | **91.1**  | **96.2**  | **99.4**  | **97.7**   |
| UMS-BERT       |       |       |       |       |
| UMS-BERT+CCE   |       |       |       |       |
| dual-bert      |       |       |       |       |
| dual-bert+CCE  |       |       |       |       |
