# Restoration-200k Dataset

* Recall Performance

* Rerank performance

| Models             | R10@1 | R10@2 | R10@5 | MRR   |
| ------------------ | ----- | ----- | ----- | ----- |
| dual-bert(seed=0; bsz=64; max_len=256,64; epoch=10, warmup_ratio=0.; lr=5e-5, grad_clip=5)      | 45.08 | 61.74 | 87.38 | 62.17 |
| bert-ft(seed=0; bsz=64; max_len=256; epoch=2, warmup_ratio=0.; lr=1e-5, grad_clip=5)            | 39.22 | 56.6 | 84.54 | 57.63 |
