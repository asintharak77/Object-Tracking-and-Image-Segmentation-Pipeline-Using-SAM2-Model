# Object Tracking and Image Segmentation Pipeline Using SAM2 Model

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
- [Usage](#usage)
  - [Initialization](#initialization)
  - [Running the Pipeline](#running-the-pipeline)
- [Project Structure](#project-structure)
- [Results and Evaluation](#results-and-evaluation)
- [Conclusion](#conclusion)
- [License](#license)

## Introduction
This project implements an object tracking and image segmentation pipeline using the SAM2 model. The pipeline processes images and video frames, tracks objects across frames, predicts segmentation masks, and evaluates the results using COCO format metrics. It is designed to be reusable and adaptable to different object categories, providing a robust solution for computer vision tasks that require precise segmentation and tracking.

## Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch
- torchvision
- NumPy
- Matplotlib
- PIL (Pillow)
- pycocotools

## Usage

### Initialization

To initialize the pipeline, create an instance of the `ImageSegmentationPipeline` class:

```python
from image_segmentation_pipeline import ImageSegmentationPipeline

model_cfg = "./model/sam2_hiera_t.yaml"
checkpoint = "./model/sam2_hiera_tiny.pt"

pipeline = ImageSegmentationPipeline(model_cfg, checkpoint)
```

### Running the Pipeline
The pipeline can be run for specific categories of objects as follows:

```python
data_dir = "./data/data_2D"
output_dir = "./outputs"
categories = ["can_chowder", "can_soymilk"]

pipeline.run_pipeline(categories, data_dir, output_dir)
```

