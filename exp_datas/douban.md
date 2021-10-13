# Douban Dataset

* Recall Performance

| Model (CPU/684208)     | Top-20 | Top-100 | Average Time(20) ms |
| ---------------------- | ------ | ------- | ------------------- |
| dual-bert-cl-flat      | 0.1124 | 0.2129  | 204.85              |
| dual-bert-cl-LSH       | 0.099  | 0.1994  | 13.51               |
| dual-bert-flat         | 0.1079 | 0.1919  | 197.65              |
| dual-bert-IVF8192,Flat | 0.057  | 0.0795  | 16.91               |
| dual-bert-IVF100,Flat  | 0.0675 | 0.1199  | 29.62               |
| dual-bert-LSH          | 0.0825 | 0.1723  | 12.05               |
| hash-bert-flat         | 0.045  | 0.1109  | 13.43               |
| hash-bert-BHash512     | 0.0435 | 0.1064  | 7.26                |

<!-- 
It should be noted that the difference between q-q and q-r is the index.
If the index is based on the responses, it is q-r matching;
If the index is based on the contexts, it is q-q matching

q-q: the constrastive loss is used for context context pair;
q-r: the constrastive loss is used for context response pair
-->
| Model (CPU/442787)     | Top-20 | Top-100 | Average Time(20) ms |
| ---------------------- | ------ | ------- | ------------------- |
| dual-bert-flat(q-r)    | 0.1229 | 0.2339  | 124.06              |
| dual-bert-fusion(q-r, ctx=full)  | 0.7826 | 0.8111  | 170.83              |
| dual-bert-fusion(q-r, ctx=1)  | 0.6087  | 0.6837 | 172.2              |
| ES(q-q) | 1.0 | 1.0 | 123.19 |
| ES(q-r) | 0.1034 | 0.1514 | 39.95 |

* Rerank performance

Comparison experments

| Original           | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ |
| SOTA               | 31.8  | 48.2  | 85.8  | 66.4  | 49.9  | 62.5   |
| HCL                | 33.0  | 53.1  | 85.8  | 68.1  | 51.4  | 63.9   |
| BERT-FP            | 32.4  | 54.2  | 87.0  | 68.1  | 51.2  | 64.4   |
| bert-ft-compare(pos_margin=0.0) | 32.05 | 54.38 | 86.57 | 67.75 | 49.93 | 63.99   |
| bert-ft-compare-plus | 31.28 | 51.8 | 82.57 | 66.2 | 48.88 | 62.12   |
| bert-ft(320, bert-fp) | 29.63 | 50.95 | 86.3 | 66.07 | 48.13 | 61.76  |
| bert-ft+compare(320, bert-fp, margin=0.55)    | 30.38 | 50.63 | 86.2 | 66.62 | 49.03 | 62.34  |
| bert-ft+compare(bert-fp, margin=0.)    | 32.91 | 54.44 | 85.71 | 68.09 | 50.97 | 64.2  |
| dual-bert(bsz=48, epoch=5, bert-fp) | 31.63 | 51.22 | 83.23 | 66.47 | 49.78  | 62.22 |
| dual-bert+compare(bsz=48, epoch=5, bert-fp, compare_turn=2, margin=0.05) | 32.2 | 53.46 | 85.21 | 67.42 | 50.07  | 63.62 |
| dual-bert+compare(bsz=48, epoch=5, bert-fp, compare_turn=2, margin=0.) | 32.42 | 53.47 | 85.02 | 67.59 | 50.52  | 63.59 |
| dual-bert+compare(bsz=48, epoch=5, bert-fp, compare_turn=1) | 31.0 | 54.47 | 85.85 | 67.51 | 49.63  | 63.42 |
| dual-bert+compare(bsz=48, epoch=5, bert-fp, compare_turn=2) | 31.0 | 54.16 | 86.28 | 67.52 | 49.63  | 63.54 |
| dual-bert+compare(bsz=48, epoch=5, bert-fp, compare_turn=2, margin=0.55) | 31.23 | 54.41 | 86.2 | 67.74 | 49.93  | 63.58 |


| Original           | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ |
| SOTA               | 31.8  | 48.2  | 85.8  | 66.4  | 49.9  | 62.5   |
| HCL                | 33.0  | 53.1  | 85.8  | 68.1  | 51.4  | 63.9   |
| bert-ft(320, bert-fp) | 29.63 | 50.95 | 86.3 | 66.07 | 48.13 | 61.76  |
| poly encoder(poly-m=32)  | 31.76 | 50.59 | 85.72 | 66.49 | 49.48 | 62.84  |
| dual-bert(bsz=32, epoch=5, poly-m=32) | 30.55 | 46.93 | 81.16 | 64.45 | 48.43  | 60.34 |
| dual-bert(bsz=32, epoch=5, bert-fp) | 31.42 | 51.6 | 83.46 | 66.41 | 49.48  | 62.22 |
| dual-bert(bsz=48, epoch=5, bert-fp) | 31.63 | 51.22 | 83.23 | 66.47 | 49.78  | 62.22 |
| dual-bert(bsz=32, epoch=5, bert-fp, proj_dim=1024) | 31.57 | 51.67 | 83.41 | 66.48 | 49.63  | 62.26 |
| dual-bert(bsz=32, epoch=5, bert-fp, lambda(gen)=0.1) | 30.95 | 50.65 | 82.95 | 65.98 | 49.03  | 61.82 |
| dual-bert(bsz=16, epoch=5, bert-post) | 27.85 | 49.26 | 85.99 | 63.88 | 44.83 | 60.73 |
| dual-bert(bsz=32, epoch=5, bert-fp-post, label-smooth, max_len=256/64) | 32.57 | 52.33 | 84.37 | 67.47 | 50.82 | 63.13 |
| dual-bert(bsz=96, epoch=5, bert-fp-post, label-smooth, max_len=256/64) | 30.65 | 51.13 | 83.05 | 66.02 | 48.88 | 61.81 |
| dual-bert-ma(bsz=32, epoch=5, bert-fp-post, label-smooth, max_len=256/64) | 32.12 | 51.63 | 82.93 | 66.86 | 50.37 | 62.54 |
| dual-bert-poly(bsz=32, epoch=5, bert-fp-post, max_len=256/64) | 31.76 | 53.25 | 84.75 | 67.51 | 50.67 | 63.27 |



| Models             | R10@1 | R10@2 | R10@5 | MRR   |  P@1  |  MAP   |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ------ |
| SOTA               | 31.8  | 48.2  | 85.8  | 66.4  | 49.9  | 62.5   |
| HCL                | 33.0  | 53.1  | 85.8  | 68.1  | 51.4  | 63.9   |
| BERT-FP            | 32.4  | 54.2  | 87.0  | 68.0  | 51.2  | 64.4   |
| bert-ft-compare    | 33.41 | 53.95 | 86.4  | 68.87 | 52.02 | 64.56  |
| dual-bert(seed=0; bsz=64; max_len=256,64; epoch=10, warmup_ratio=0.; lr=5e-5, grad_clip=5)      | 33.13  | 53.99  | 86.0  | 68.41  | 51.27  | 64.28   |
| dual-bert+compare(seed=0; bsz=64; max_len=256,64; epoch=10, warmup_ratio=0.; lr=5e-5, grad_clip=5; compare_turn=2; margin=0.05)      | 33.51  | 53.41  | 87.05  | 68.59  | 51.87  | 64.76   |
| bert-ft(seed=0; bsz=64; max_len=256; epoch=2, warmup_ratio=0.; lr=1e-5, grad_clip=5)            | 31.1  | 54.52  | 86.36  | 67.72  | 49.93  | 63.76   |
