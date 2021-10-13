# LCCC Dataset

* recall performance

| Models(CPU/394740/full-4769532)    | Top-20 | Top-100 | Time   |
| ---------- | ----- | ----- | --- |
| dual-bert-flat  | 0.105 | 0.195 | 112.39 | 
| dual-bert-fusion-flat  | 0.571 | 0.624  | 110.5 |
| dual-bert-full-LSH  | 0.038 | - | 68.76 | 
| dual-bert-full-fusion-LSH  | 0.417 | 0.475 | 68.92 |
| ES-full(q-q) |  |  |  |
| ES-full(q-r) |  |  |  |
| ES(q-q) | 0.979 | 0.995 | 16.96 |
| ES(q-r) | 0.051 | 0.099 | 9.4 |

* rerank performance

| Models    | R10@1 | R10@2 | R10@5 | MRR   |
| --------- | ----- | ----- | ----- | ----- |
| dual-bert | 40.5 | 75.0 | 92.8 | 63.88 |
| dual-bert-fusion | 40.7 | 73.9 | 92.5 | 63.77 |
