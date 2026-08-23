# Tensor Steganography Demo

A small PyTorch research demonstration showing how data can be
encoded into the least-significant bits of neural-network weights
while preserving approximately the same numerical parameter values.

## Purpose

This demonstrates:

- Neural-network weights stored as float32 tensors
- Least-significant-bit modification
- Encoding text into model parameters
- Recovering embedded data
- Comparing original and modified model weights

This demo uses only harmless text data and is intended for
educational security research.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Usage

Train a baseline MNIST model:

```bash
python /src/train.py
```

Embed the demonstration payload:

```bash
python src/embed.py
```

Compare the models:

```bash
python src/compare.py
```

| Generated datasets and model files are excluded from version control.











