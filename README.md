# Continuous Wasserstein-2 Benchmark
This is the official `Python` implementation of the [NeurIPS 2021](https://nips.cc/) paper **Do Neural Optimal Transport Solvers Work? A Continuous Wasserstein-2 Benchmark** (paper on [arxiv](https://arxiv.org/abs/2106.01954)) by [Alexander Korotin](https://scholar.google.ru/citations?user=1rIIvjAAAAAJ&hl=en), [Lingxiao Li](https://scholar.google.com/citations?user=rxQDLWcAAAAJ&hl=en), [Aude Genevay](https://scholar.google.com/citations?user=SryRaIMAAAAJ), [Justin Solomon](https://scholar.google.com/citations?user=pImSVwoAAAAJ), [Alexander Filippov](https://scholar.google.com/citations?user=fY5epnkAAAAJ) and [Evgeny Burnaev](https://scholar.google.ru/citations?user=pCRdcOwAAAAJ&hl=ru).

The repository contains a set of continuous benchmark measures for testing optimal transport solvers for quadratic cost (Wasserstein-2 distance), the code for optimal transport solvers and their evaluation.

## Citation
```
@article{korotin2021neural,
  title={Do Neural Optimal Transport Solvers Work? A Continuous Wasserstein-2 Benchmark},
  author={Korotin, Alexander and Li, Lingxiao and Genevay, Aude and Solomon, Justin and Filippov, Alexander and Burnaev, Evgeny},
  journal={arXiv preprint arXiv:2106.01954},
  year={2021}
}
```

## Pre-requisites
The implementation is GPU-based. Single GPU (~GTX 1080 ti) is enough to run each particular experiment. Tested with

`torch==1.3.0 torchvision==0.4.1`

The code might not run as intended in newer `torch` versions.

## Related repositories
- [Repository](https://github.com/iamalexkorotin/Wasserstein2GenerativeNetworks) for [Wasserstein-2 Generative Networks](https://openreview.net/pdf?id=bEoxzW_EXsa) paper.
- [Repository](https://github.com/iamalexkorotin/Wasserstein2Barycenters) for [Continuous Wasserstein-2 Barycenter Estimation without Minimax Optimization](https://arxiv.org/abs/2102.01752) paper.
- [Repository](https://github.com/lingxiaoli94/CWB) for [Continuous Regularized Wasserstein Barycenters](https://proceedings.neurips.cc/paper/2020/file/cdf1035c34ec380218a8cc9a43d438f9-Paper.pdf) paper.
- [Repository](https://github.com/PetrMokrov/Large-Scale-Wasserstein-Gradient-Flows) for [Large-Scale Wasserstein Gradient Flows](https://arxiv.org/abs/2106.00736) paper.

## Loading Benchmark Pairs
```python
from src import map_benchmark as mbm

# Load benchmark pair for dimension 16 (2, 4, ..., 256)
benchmark = mbm.Mix3ToMix10Benchmark(16)
# OR load 'Early' images benchmark pair ('Early', 'Mid', 'Late')
# benchmark = mbm.CelebA64Benchmark('Early')

# Sample 32 random points from the benchmark measures
X = benchmark.input_sampler.sample(32)
Y = benchmark.output_sampler.sample(32)

# Compute the true forward map for points X
X.requires_grad_(True)
Y_true = benchmark.map_fwd(X, nograd=True)
```
## Repository structure
All the experiments are issued in the form of pretty self-explanatory jupyter notebooks (`notebooks/`). Auxilary source code is moved to `.py` modules (`src/`). Continuous benchmark pairs are stored as `.pt` checkpoints (`benchmarks/`).

## Evaluation of Existing Solvers
We provide all the code to evaluate existing dual OT solvers on our benchmark pairs.

### Testing Existing Solvers On High-Dimensional Benchmarks
- `notebooks/MM_test_hd_benchmark.ipynb` -- testing \[MM\], \[MMv2\] solvers and their reversed versions
- `notebooks/MMv1_test_hd_benchmark.ipynb` -- testing \[MMv1\] solver
- `notebooks/MM-B_test_hd_benchmark.ipynb` -- testing \[MM-B\] solver
- `notebooks/W2_test_hd_benchmark.ipynb` -- testing \[W2\] solver and its reversed version
- `notebooks/QC_test_hd_benchmark.ipynb` -- testing \[QC\] solver
- `notebooks/LS_test_hd_benchmark.ipynb` -- testing \[LS\] solver

### Testing Existing Solvers On Images Benchmark Pairs (CelebA 64x64 Aligned Faces)
- `notebooks/MM_test_images_benchmark.ipynb` -- testing \[MM\] solver and its reversed version
- `notebooks/W2_test_images_benchmark.ipynb` -- testing \[W2\]
- `notebooks/MM-B_test_images_benchmark.ipynb` -- testing \[MM-B\] solver
- `notebooks/QC_test_images_benchmark.ipynb` -- testing \[QC\] solver

\[LS\], \[MMv2\], \[MMv1\] solvers are not considered in this experiment.

### Generative Modeling by Using Existing Solvers to Compute Loss
**Warning:** training may take several days before achieving reasonable FID scores!
- `notebooks/MM_test_image_generation.ipynb` -- generative modeling by \[MM\] solver or its reversed version
- `notebooks/W2_test_image_generation.ipynb` -- generative modeling by \[W2\] solver

For \[QC\] solver we used the code from the [official WGAN-QC repo](https://github.com/harryliew/WGAN-QC).

## Training Benchmark Pairs From Scratch
This code is provided for completeness and *is not intended to be used to retrain existing benchmark pairs*, but might be used as the base to train new pairs on new datasets. High-dimensional benchmak pairs can be trained from scratch. Training images benchmark pairs requires generator network checkpoints. We used [WGAN-QC](https://github.com/harryliew/WGAN-QC) model to provide such checkpoints.
- `notebooks/W2_train_hd_benchmark.ipynb` -- training high-dimensional benchmark bairs by \[W2\] solver
- `notebooks/W2_train_images_benchmark.ipynb` -- training images benchmark bairs by \[W2\] solver

## Credits
- [Weights & Biases](https://wandb.ai) developer tools for machine learning;
- [CelebA page](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) with faces dataset and [this page](https://www.kaggle.com/jessicali9530/celeba-dataset) with its aligned 64x64 version;
- [pytorch-fid repo](https://github.com/mseitzer/pytorch-fid) to compute [FID](https://arxiv.org/abs/1706.08500) score;
- [UNet architecture](https://github.com/milesial/Pytorch-UNet) for transporter network;
- [ResNet architectures](https://github.com/harryliew/WGAN-QC) for generator and discriminator;
