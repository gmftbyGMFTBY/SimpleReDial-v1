# E-Commerce Dataset

* recall performance

| Model (CPU/109105)     | Top-20 | Top-100 | Average Time(20) ms |
| ---------------------- | ------ | ------- | ------------------- |
| dual-bert-flat(q-r)    |  0.567 | 0.791   | 29.51               |
| dual-bert-fusion(q-r)  |  0.561 | 0.645   | -              |

* rerank performance

| Original       | R10@1 | R10@2 | R10@5 | MRR    |
| -------------- | ----- | ----- | ----- | ------ |
| SOTA           | 77.6  | 91.9  | 99.1  | -      |
| BERT-FP        | 87.0  | 95.6  | 99.3  | -      |
| dual-bert      | 91.7  | 96.0  | 99.2  | 94.85  |
| dual-bert(dual-bert-pt)      | 90.1  | 96.0  | 98.9  | 94.03  |
| bert-ft-compare    | 94.2  | 97.8  | 99.5  | 96.52  |
| bert-ft-compare-plus    | 95.8  | 98.8  | 99.9  | 97.67  |
| bert-ft-compare(epoch=10, super_hard_negative) | 93.1 | 98.5 | 99.9 | 96.22 |
| bert-ft-compare(epoch=10, super_hard_negative, margin=0.1) | 95.6 | 99.1 | 100 | 97.58 |
| dual-bert          | 91.7  | 96.0  | 99.2  | 94.85  |
| dual-bert+compare  | 92.0  | 96.6  | 99.6  | 95.23  |
| dual-bert+compare(super_hard_negative, margin=0.1, compare_turn_num=2)  | 90.3  | 96.4  | 99.7  | 94.32  |
| dual-bert+compare(super_hard_negative, margin=0.0, compare_turn_num=2)  | 90.9  | 96.5  | 99.7  | 94.65  |
| bert-ft            | 83.4  | 94.4  | 99.4  | 90.43  |
| bert-ft+compare(super_hard_negative, margin=0.0)            | 91.8  | 97.5  | 99.7  | 95.35  |
| bert-ft+compare(with not sure)    | 93.7  | 98.0  | 99.6  | 96.34  |
| bert-ft+compare(without not sure)    | 92.7  | 97.3  | 99.6  | 95.72  |


| Models         | R10@1 | R10@2 | R10@5 | MRR    |
| -------------- | ----- | ----- | ----- | ------ |
| SOTA           | 77.6  | 91.9  | 99.1  | -      |
| BERT-FP        | 87.0  | 95.6  | 99.3  | -      |
| dual-bert(seed=0; grad_clip=5; lr=5e-5; bsz=64; warmup_ratio=0; epoch=10; max_len=256,64)        | 92.5  | 97.0  | 99.4  | 95.49      |
