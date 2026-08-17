# PyTorch Deep Learning & Computer Vision Foundations

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA Ready](https://img.shields.io/badge/CUDA-Acceleration-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade, end-to-end deep learning and computer vision repository covering fundamental tensor mathematics, custom neural architectures, non-linear classification, convolutional vision pipelines (TinyVGG), dynamic data augmentation, device-agnostic GPU acceleration and modular CLI-driven training engines.

---

## 📌 Repository Structure

```text
pytorch-deep-learning-foundations/
├── 00_pytorch_fundamentals/
│   ├── 00_pytorch_fundamentals.ipynb
│   └── 00_pytorch_fundamentals_solutions.ipynb
├── 01_pytorch_workflow/
│   ├── 01_pytorch_workflow.ipynb
│   └── 01_pytorch_workflow_solutions.ipynb
├── 02_pytorch_neural_network_classification/
│   ├── 02_pytorch_classification.ipynb
│   └── 02_pytorch_classification_solutions.ipynb
├── 03_pytorch_computer_vision/
│   ├── 03_pytorch_computer_vision.ipynb
│   └── 03_pytorch_computer_vision_solutions.ipynb
├── 04_pytorch_custom_datasets/
│   ├── 04_pytorch_custom_datasets.ipynb
│   └── 04_pytorch_custom_datasets_solutions.ipynb
├── 05_pytorch_going_modular/
│   ├── 05_pytorch_going_modular.ipynb
│   ├── 05_pytorch_going_modular_solutions.ipynb
│   └── going_modular/
│       ├── data_setup.py
│       ├── engine.py
│       ├── model_builder.py
│       ├── train.py
│       └── utils.py
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```
---
## 🔬 Module Breakdown

| Module | Core Concepts & Systems Implemented | Key APIs / Mathematical Techniques |
| :--- | :--- | :--- |
| [**`00_pytorch_fundamentals`**](./00_pytorch_fundamentals/) | Tensor initialization, memory layouts, dimensional manipulation, device-agnostic execution (CPU/CUDA), batch matrix multiplication. | `torch.matmul`, `torch.squeeze`, `torch.permute`, `torch.manual_seed` |
| [**`01_pytorch_workflow`**](./01_pytorch_workflow/) | End-to-end regression pipeline, `nn.Module` subclassing, forward computational graphs, parameter updates, serialization. | `nn.Linear`, `nn.L1Loss`, `torch.optim.SGD`, `state_dict` |
| [**`02_pytorch_neural_network_classification`**](./02_pytorch_neural_network_classification/) | Non-linear geometric separation (Circles, Blobs), logit-to-probability mapping, numerical stability under extreme loss bounds. | `BCEWithLogitsLoss`, `CrossEntropyLoss`, `nn.ReLU`, `torchmetrics` |
| [**`03_pytorch_computer_vision`**](./03_pytorch_computer_vision/) | Multi-class image classification (FashionMNIST), NCHW tensor conventions, 2D convolutional filter design, max pooling downsampling. | `nn.Conv2d`, `nn.MaxPool2d`, `torchvision.transforms`, `TinyVGG` |
| [**`04_pytorch_custom_datasets`**](./04_pytorch_custom_datasets/) | Custom disk ingestion pipelines (`Dataset`/`DataLoader`), stochastic data augmentations (`TrivialAugmentWide`), transfer learning adaptation. | `ImageFolder`, `__getitem__` override, `torchvision.models` |
| [**`05_pytorch_going_modular`**](./05_pytorch_going_modular/) | Refactoring notebook code into production-ready, reusable Python engines with parameterizable CLI entry points. | `argparse`, modular engine abstractions, automated asset caching |

---
### ⚙️ Modular Engine Architecture (going_modular/)
The repository transitions research prototypes into a clean, decoupled production package:

    going_modular/
    ├── data_setup.py      # Ingestion logic, directory parsing, transformations, and DataLoader pipeline generation
    ├── engine.py          # Device-agnostic train_step, test_step, and multi-epoch loops
    ├── model_builder.py   # Parameterized CNN architecture (TinyVGG) definition
    ├── train.py           # Making executable CLI orchestration script tying data loading, model instantiation, training loops with dynamic hyperparameters
    └── utils.py           # Reusable utilities for saving model checkpoints and directory setup.

### Modular CLI Usage
**1. Dataset Ingestion**
```bash
python going_modular/data_setup.py
```
**2. Model Training with Hyperparameter Overrides**
```bash
python going_modular/train.py \
    --model_name tinyvgg_food101.pth \
    --num_epochs 20 \
    --batch_size 32 \
    --hidden_units 64 \
    --learning_rate 0.001
```
**3. Single-Sample Inference**
```bash
python going_modular/predict.py \
    --model_path models/tinyvgg_food101.pth \
    --image data/pizza_steak_sushi/test/pizza/1152100.jpg
```
---
## 🛠️ Key Architectural Decisions

* **Numerical Stability via Logit-Level Loss Formulation:**  
  Replaced raw `BCELoss(Sigmoid(x))` with `BCEWithLogitsLoss` to combine the sigmoid layer and cross-entropy step into a single mathematically fused operation, preventing vanishing gradients and `NaN` instability during backpropagation.

* **Device-Agnostic Acceleration:**  
  Standardized tensor and model transitions to automatically detect and leverage CUDA environments while maintaining seamless fallback to CPU execution:

  ```python
  device = "cuda" if torch.cuda.is_available() else "cpu"
  ```
* **Decoupled CLI & Modular Script Architecture:**
  Transitioned exploratory Jupyter prototypes into an isolated, reusable production package (going_modular/). Separated dataset ingestion, model definition, training/evaluation loops, checkpoint persistence, and inference into independent, parameterizable CLI modules driven by `argparse`.

* **Zero-Artifact Commit Policy:**
  All evaluation metrics, decision boundary surfaces, confusion matrices, and loss curves are executed and embedded directly within the .ipynb cells to keep the repository lightweight and eliminate out-of-sync visual file dependencies.

---
## 🚀 Quickstart & Environment Setup

### Prerequisites
* Python 3.10 or higher
* NVIDIA GPU with CUDA support (optional, but recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/Subhrajyoti8520/pytorch-deep-learning-foundations.git
cd pytorch-deep-learning-foundations

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
**Launch Jupyter:**
```bash
jupyter lab
# or
jupyter notebook
```
---
## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.


