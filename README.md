# embedding-metrics
This is an implementation of metrics for unsupervised embedding quality evaluation on Jax, based on the paper "Unsupervised Embedding Quality Evaluation".

The original code in numpy can be found here: https://github.com/google-research/google-research/tree/master/graph_embedding/metrics

With the application of Jax, computations were able to be accelerated ~2x on CPU and ~15x on GPU on my hardware.

## Citation
Tsitsulin, A., Munkhoeva, M., & Perozzi, B. (2023). Unsupervised Embedding Quality Evaluation. [Paper](https://arxiv.org/abs/2305.16562)
