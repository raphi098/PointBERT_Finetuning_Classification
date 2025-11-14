# PointBERT Classification

This repository provides functionality to finetune a PointBERT backbone on a classification task. It is based on the original implementation of [Point-BERT](https://github.com/Julie-tang00/Point-BERT).

The project uses a stack of **PyTorch Lightning**, **Hydra**, and **MLflow**. Furthermore, a **Dockerfile** is provided to avoid the pain of installing all extensions manually. The extensions **KNN**, **EMD**, **Pointnet_ops**, and **ChamferDistance** were adapted to produce `.so` files so they can be used in a multi-stage Docker build. This helps keep the final image size small. The current version is for Linux only but can also be used on Windows via WSL and Docker

---

## Installation

### Option 1: Docker (recommended)

```bash
# Build the image
docker build -t pointbert .

# Run the container on Linux/WSL (bind current directory into /app)
docker run --gpus all -it --attach type=bind,src=$(pwd),target=/app pointbert
```

### Option 2: Virtual environment or Conda

If you do not want to use Docker, you can install everything manually in a local environment.

1. Make sure the **CUDA toolkit** is installed and your **PyTorch** installation matches the toolkit version.
2. Install PyTorch (recommended version):

   ```bash
   pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1        --index-url https://download.pytorch.org/whl/cu126
   ```

3. Install the custom CUDA extensions:

   ```bash
   bash install.sh
   ```

4. Install the remaining Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

A training or test run can be started by running:

```bash
python main.py
```

The exact behavior (for example, training only, testing only, or both) is controlled through the Hydra configuration used in `main.py` and the files in the `config/` directory.

---

## Repository Structure

### `config/`

Configuration files are located in the `config` folder.

- The main idea of this implementation is to expose the most important variables in `main.py`, so you only need to adapt this file for different training runs.
- In `config/model/pointbert.yaml`, only the **backbone name** should normally be changed.
- All other parameters should be configured in `main.py`.

### `checkpoints/`  *(must be created manually)*

This folder must contain the pretrained checkpoint that is referenced in `config/model/pointbert.yaml`.

Example:

```text
checkpoints/
└── pointbert_backbone.ckpt
```

### `dataset/`  *(must be created manually)*

The `dataset` folder contains the dataset you are working with.  
The recommended structure, which is also used in the custom dataset class, is:

```text
dataset/
└── <dataset_name>/
    ├── <category_folder1>/
    ├── <category_folder2>/
    ├── ...
    ├── train.txt
    ├── val.txt
    ├── test.txt
    └── cat_dict.json
```

#### Text files: `train.txt`, `val.txt`, `test.txt`

These files contain the **relative path** to each sample, one per line (separated by `\n`).

Example:

```text
1gliedrig/1gliedrig_(5933).npy
modellguss/modellguss_(1611).npy
2gliedrig/2gliedrig_(12002).npy
1gliedrig/1gliedrig_(3726).npy
...
```

#### Category mapping: `cat_dict.json`

This file contains the mapping `category_name: category_id`.

Example:

```json
{
    "1gliedrig": 0,
    "2gliedrig": 1
}
```
