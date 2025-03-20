# Enhancing Prototypical Networks with Feature Masking

This repository contains experiments and tests on feature masking applied to prototypical networks and other few-shot learning paradigms using the `easyfsl` library. The goal is to explore how feature masking can improve the performance of few-shot learning models.

## Getting Started

### Prerequisites

- Python 3.x

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/carson-cheng/fsl_feature_masking.git
   cd fsl_feature_masking
   ```
2. Install the required dependencies:
   ```
   cd easy-few-shot-learning
   pip install .
   ```
### Running the Experiments

#### Single Evaluation on Prototypical Network

To run a single evaluation on the Prototypical Network, use the following command:

```
python3 fewshot.py
```

#### Fine-Tuning Evaluation

To run a fine-tuning evaluation where the vision backbone is also fine-tuned, use the following command:

```
python3 2_fewshot.py
```

#### Batched-Up Evaluation

To run the batched-up evaluation across multiple datasets (CUB -> mini-ImageNet -> Aircraft -> CIFAR -> Flowers), follow these steps:

Download the required datasets using kagglehub:
```
root = kagglehub.dataset_download("wenewone/cub2002011")
root = kagglehub.dataset_download("arjunashok33/miniimagenet")
```
Run the batched-up evaluation script:

```
python3 loop_fewshot.py
```

# Datasets
The following datasets are used in this repository:

CUB-200-2011: A dataset of 200 bird species with 11,796 images.

mini-ImageNet: A smaller version of the ImageNet dataset, with 200 classes across 60,000 images.

Aircraft: A dataset of 100 aircraft models with 10,000 images. (the script only uses the test set of 3333 images)

CIFAR: A dataset of 60,000 32x32 color images in 10 classes. (the script only uses the test set of 10000 images)

Flowers: A dataset of 102 flower categories with 8,189 images. (the script only uses the test set of 6149 images)