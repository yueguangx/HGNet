# HGNet
This project is based on the paper "HGNet: High-Order Spatial Awareness Hypergraph and Multi-Scale Context Attention Network for Colorectal Polyp Detection". It implements and optimizes the proposed HGNet model for colorectal polyp detection, leveraging high-order spatial awareness and multi-scale context attention mechanisms. The goal is to improve detection accuracy and efficiency in medical image analysis.



# Installation 
Follow these steps to set up and run the HGNet project on your local machine:

**Clone the repository**:
   ```bash
   git clone https://github.com/yueguangx/HGNet.git
   cd HGNet
   conda create -n HGNet python=3.8
   conda activate HGNet
   pip install -r requirements.txt


# **Dataset Preparation**
To train and evaluate the HGNet model, it's essential to organize your dataset in the COCO format. Below are the steps to prepare your dataset:
   ```bash
**Directory Structure**：

Organize your dataset with the following structure:

```plaintext
├── coco
│   ├── images
│   │   ├── train2017
│   │   └── val2017
│   └── labels
│       ├── train2017
│       └── val2017

