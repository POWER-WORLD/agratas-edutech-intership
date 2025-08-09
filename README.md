# Image Matching with Transfer Learning (Internship Project)
**Topic:** Solving an Image Matching Problem Using Transfer Learning  
**Internship:** Agratas EduTech (2 June 2025 - 2 August 2025)

## Overview
This project implements an image matching pipeline using transfer learning with a Siamese-style architecture built on a pretrained ResNet50 backbone (PyTorch). The repository contains:
- `train.py` : Training and evaluation entrypoint for a Siamese model.
- `model.py` : Model definitions (Siamese network + embedding extractor from pretrained ResNet).
- `utils.py` : Dataset and helper utilities (pair generation, transforms, metrics).
- `requirements.txt` : Python dependencies
- Example usage instructions below.

## Example usage (local)
1. Prepare dataset in the following structure (or adapt the dataset loader):
   ```
   dataset/
     class1/
       img1.jpg
       img2.jpg
     class2/
       img3.jpg
       ...
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Train:
   ```
   python train.py --data_dir ./dataset --epochs 10 --batch_size 32 --output_dir ./outputs
   ```

## Notes
- This code is written to be clear and educational. For large datasets or production use, adapt data pipelines and training routines to use better logging, checkpointing, and distributed training as needed.
