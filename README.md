# LUNA Data Challenge - Tumor Malignancy Prediction

## Overview

This project focuses on the processing of RAW CT scan data to facilitate the identification and classification of lung nodules. The workflow goes as follows:
1. Data loading of the <strong>.mhd</strong> and <strong>.raw</strong> file. 
2. Segmentation 
3. Grouping 
4. Nodule classification (Nodule, or Non-Nodule)
5. Nodule analysis and diagnosis (Malignant/Benign)

## Usage

To get a deeper understanding of specific aspects of the project, consider exploring the `read_info` folder which contains the following notebooks:

  - `generate_annotations_with_malignancy`: Learn how malignancy labels are generated for the project.
  
- `convolutions`: Understand the concept of 2D convolution kernels.
 Dissect the output and compare it to the input as a general example to comprehend CNNs.

- `data`: Object-oriented implementation of data preprocessing.
Dive into the preprocessing of data as part of the project using an object-oriented implementation. Note that the LunaData_v2 class for the dataset is the one that is utilized for training in the final project.

- `pipeline`: Comprehensive breakdown of the training workflow, model architecture, GPU-based data augmentation, and metric logging etc. Explore the step-by-step process of the project pipeline.



# Material 

The Manning site for the book is: https://www.manning.com/books/deep-learning-with-pytorch

The book can also be purchased on Amazon: https://amzn.to/38Iwrff (affiliate link; as per the rules: "As an Amazon Associate I earn from qualifying purchases.")


# Data 
The data can be accessed [here](https://luna16.grand-challenge.org/Data/). For our implementation, we only used subset0.
