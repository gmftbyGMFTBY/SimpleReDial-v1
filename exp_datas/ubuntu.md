# Ubuntu V1 Dataset

* rerank performance

| Original       | R10@1 | R10@2 | R10@5 | R2@1   |
| -------------- | ----- | ----- | ----- | ------ |
| SOTA           | 0.884 | 0.946 | 0.990 | 0.975  |
| HCL            | 86.7 | 94.0 | 99.2 | 97.7  |
| BERT-FP        | 91.1  | 96.2  | 99.4  | 0.977  |
| dual-bert(lr=5e-5; epoch=10; seed=0; warmup=0; max_len=256,64) | 88.57 | 95.06 | 99.09 | - |
| bert-ft(lr=5e-5; epoch=5; seed=0; warmup=0; max_len=256,64) | 90.16 | 95.82 | 99.25 | - |

    1000 test samples for fast iterations

| Model             | R10@1 | R10@2 | R10@5 | MRR    |
| ----------------- | ----- | ----- | ----- | ------ |
| BERT-FP           | 87.7  | 93.8  | 98.6  | 92.31  |
| bert-ft-compare-plus   | 92.8  | 98.2  | 100  | 96.07  |
| bert-ft-compare(pos=0.15)   | 88.7  | 94.1  | 97.1  | 92.69  |
| bert-ft-compare(pos=0.15, super-hard-negative)   | 89.8  | 95.9  | 99.3  | 93.91  |
| bert-ft           | 89.7  | 96.0  | 99.3  | 93.92  |
| bert-ft(lr=5e-5; warmup_ratio=0; grad_clip=5; seed=0; epoch=5)  | 91.3 | 96.7  | 99.5  | 94.85  |
| bert-ft+compare-plus(margin=-0.1)   | 87.7  | 94.6  | 99.3  | 92.63  |
| bert-ft+compare(margin=-0.1)   | 87.7  | 94.6  | 99.3  | 92.63  |
| bert-ft+compare(margin=0.0)   | 87.1  | 93.9  | 99.1  | 92.14  |
| bert-ft+compare(margin=0.1)   | 86.2  | 93.6  | 98.9  | 91.59  |
| bert-ft+compare   | 89.1  | 95.1  | 99.0  | 93.38  |
| bert-ft+compare(pos_margin=-0.2)   | 89.0  | 94.9  | 99.0  | 93.3  |
| dual-bert         | 86.8  | 94.5  | 98.5  | 91.98  |
| dual-bert(lr=5e-5; warmup_ratio=0; epoch=10; grad_clip=5; seed=0)  | 89.0 | 95.6  | 98.8  | 94.41  |
| dual-bert+compare(pos_margin=0.1) | 87.2  | 94.0  | 98.3  | 92.1  |
| dual-bert+compare(pos_margin=-0.1) | 88.2  | 94.6  | 98.3  | 92.72  |
| dual-bert+compare(pos_margin=-0.2) | 87.7  | 94.1  | 98.1  | 92.33  |

* recall performance

    Because of the very large test set, we use the LSH other than the Flat

| Originali (545868)       | Top-20 | Top-100 | Time |
| -------------- | ----- | ----- | ------ |
| dual-bert-LSH | 0.1374 | 0.2565 | 8.59  |
| dual-bert-fusion-LSH | 0.7934 | 0.8147 | 7.89  |
| ES(q-q) | 0.0101 | 0.0202 | 22.13 |
| ES(q-r) | 0.0014 | 0.0083 | 9.79 |
