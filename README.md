# Deep Learning

A curated collection of classic ML and deep learning assignments and mini-projects. The repo spans classical methods (kNN, SVM, CART) and deep models (MLP from scratch, CNNs on CIFAR-10, RNNs, VAEs), plus a small tracking project (DeepSORT/ByteTrack).

## Repository Structure

- `Classic ML/`
  - `CART/`
    - `CART.ipynb`: Decision trees (CART) experiments
  - `kNN/`
    - `knn.ipynb`: k-Nearest Neighbors notebook
    - `k_nearest_neighbor.py`, `data_utils.py`: helper modules
    - `datasets/`: dataset scripts/assets (e.g., `get_datasets.sh`)
  - `SVM/`
    - `SVM.ipynb`: Support Vector Machine experiments on `Liver_Disease.csv`

- `MLP/`
  - `MLP Scratch/`
    - `hw2_1.ipynb`: MLP built from scratch
    - `layers.py`, `optimizer.py`, `solver.py`: core MLP components
    - `data/MNIST/raw/`: MNIST data files (idx format)
  - `Loss on MLP/`
    - `DL-HW2-P2.ipynb`: MLP loss exploration and decision boundaries

- `CNN/`
  - `P1.ipynb`, `P2.ipynb`, `P3.ipynb`: CNN assignments on CIFAR-10/Flower102
  - `data/cifar-10-batches-py/`: CIFAR-10 Python-format batches (and `cifar-10-python.tar.gz`)
  - `utils/utils.py`: common utilities

- `RNN/`
  - `DeepLearning_HW4_p1.ipynb`, `DeepLearning_HW4_p2.ipynb`, `DeepLearning_HW4_p3.ipynb`: RNN tasks

- `VAE/`
  - `Q1_VAE_CVAE.ipynb`: Variational Autoencoder and Conditional VAE
  - `Q2_VQ_VAE.ipynb`: Vector-Quantized VAE

- `Project/`
  - `phase1.ipynb`, `phase2.ipynb`: MOT tracking phases
  - Precomputed result videos: `bytetrack_tracking_result.mp4`, `mot_tracking_results.mp4`, etc.

PDFs for each homework/project are included alongside the code for reference.
