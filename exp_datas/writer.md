# Writer Dataset

<!-- Inference -->
* Recall performance

| Models                              | Top-20 | Top-100 | Time Cost   |
| ----------------------------------- | ------ | ------- | ----------- |
| dual-bert-gray-writer-LSH           | 0.126  | 0.193   | 529.10      |
| dual-bert-fusion-gray-writer-LSH    |        |         |             |
| hash-bert-gray-writer-BHash512      |        |         |             |

* Rerank performance

| Models                              | R10@1 | R10@2 | R10@5 | MRR   |
| ----------------------------------- | ----- | ----- | ----- | ----- |
| bert-base-chinese (dual-bert-gray-writer) | 66.12 | 79.9 | 94.37 | 77.88 |
| bert-base-chinese (hash-bert-gray-writer) | 54.16 | 75.72 | 94.78 | 71.13 |
| bert-base-chinese (dual-bert|g=2)   | 56.32 | 73.06 | 92.67 | 71.22 |
