# Contrails Detection

## Abstract
Contrails, the line-shaped ice clouds formed by aircraft, represent a substantial contributor to aviation-induced climate change. Addressing this environmental concern necessitates effective contrail avoidance strategies, which offer a potentially economical means to mitigate the overall climate impact of aviation. To develop and evaluate such contrail avoidance systems, the implementation of an automated contrail detection system is crucial. In this paper, we introduce two deep learning techniques designed for contrail detection, utilizing data from the GOES-16 Advanced Baseline Imager (ABI). These techniques leverage temporal context to enhance detection accuracy, providing a more robust framework for identifying contrails in satellite imagery. The proposed methods contribute to the ongoing efforts in developing sustainable aviation practices by offering advanced tools for precise contrail detection, thereby facilitating the implementation of effective contrail avoidance strategies and reducing the aviation industry's environmental footprint.

## Dataset
The dataset utilized in this research originates from the GOES-16 Advanced Baseline Imager (ABI) imagery dataset, as made publicly available in the paper "OpenContrails: Benchmarking Contrail Detection on GOES-16 ABI" by Joe Yue-Hei Ng et al. This dataset serves as a foundational resource for developing and evaluating contrail detection models, contributing to the broader efforts aimed at addressing the environmental impact of contrails on climate change.

The dataset comprises 20,544 examples in the training set and 1,866 examples in the validation set. Random partitioning was applied, with scenes identified as likely to have contrails by Google Street View included only in the training set. Of the training examples, 9,283 contain at least one annotated contrail. Approximately 1.2% of the pixels in the training set are labeled as contrails. The dataset exhibits a diverse array of times and locations, as visually depicted in Figures 5 and 6, with non-uniform distribution due to strategic sampling for increased contrail representation. Provided in TFRecord format, each example includes a 256 × 256 image with multiple GOES-16 ABI brightness temperatures and brightness temperature differences.

### Dataset preparation for training Simple U-Net Model

To enhance the dataset for the purposes of contrail detection, a transformation process was employed, as detailed in the "False Color Dataset Generation Notebook". This notebook outlines the steps taken to create a specialized dataset for training the Unet model.

The transformation process involves several key steps:
1.	Data Frame Creation: Established train and validation data frames to organize record IDs for each image.
2.	Image Saving: Saved labeled frames, human pixel masks, ash color images, and mask labels as numpy arrays, stored in single files for streamlined model access.
3.	Normalization and False Color Image Generation: Normalized data ranges and generated false color images by combining specific bands, enhancing contrail visibility.
4.	Data Storage: Stored resulting numpy files, containing ash color images and mask labels, in a designated directory in float16 dtype for optimized data size.
5.	Utilization in Unet Model: Employed transformed numpy files as primary input data for training the Unet model, leveraging its effectiveness in image segmentation for contrail detection.
    The specific bands used in the "False Color Dataset Generation Notebook" were selected to justify the emphasis on ice-clouds, to accurately detect contrails. To facilitate detection throughout both day and night, this imagery is transformed to an "ash" false color scheme. This scheme combines three longwave GOES-16 brightness temperatures, representing the 12μm, the difference between 12μm and 11μm, and the difference between 11μm and 8μm, respectively. This color scheme aids in identifying contrails by emphasizing ice-clouds as darker colors.

### Dataset preparation for training U-Net Variant Model
1.	BTD Calculation:
○	Calculate Brightness Temperature Differences (BTD) by subtracting 10.35 μm from 12.3 μm for GOES-16.
○	Isolate optically thin cirrus clouds and eliminate interference.
2.	Contrail Identification:
○	Identify days and regions with contrail occurrence using BTD images.
○	Download GOES-16 data with goes2go.
3.	Image Generation:
○	Process downloaded files with netCDF4.
○	Generate the final image by subtracting 10.35 μm from 12.3 μm.
4.	Projection Conversion:
○	Convert the image to a local projection using pyproj.
5.	Contrail Tracing and Mask Generation:
○	Use GIMP for tracing contrails and generating a mask.
○	Strokes of about two pixels represent contrail paths.
These steps enhance features for accurate contrail detection in satellite imagery during subsequent model training.

## Evaluation Metrics
Dice Coefficient: This coefficient serves to compare pixel-wise agreement between predicted contrail segmentation and the corresponding ground truth. 
Intersection over Union (IoU): The IoU is calculated as the ratio of the area of intersection between the predicted and ground truth regions to the area of their union. 

## Models

Model 1 : Simple U-Net Architecture
1.	Encoder:
○	Utilizes convolutional layers with increasing filters and 3x3 kernel sizes.
○	Applies Rectified Linear Unit (ReLU) activation functions.
○	Employs MaxPooling layers for spatial downsampling.
2.	Bottleneck:
○	Features a bottleneck layer with two convolutional layers to capture high-level features.
3.	Decoder:
○	Implements Transposed Convolutional layers for upsampling.
○	Incorporates skip connections by concatenating corresponding feature maps from the encoder.
○	Utilizes Convolutional layers with decreasing filters for refining segmentation details.
○	Applies ReLU activation functions.
4.	Output Layer:
○	Comprises a convolutional layer with sigmoid activation for binary segmentation (contrail or non-contrail).

Model 2 : U-Net variant with Hough Transform
The U-Net variant employed in our model for contrail segmentation combines the strengths of the U-Net architecture and ResNet (Residual Network) [12]. Here's a simplified overview of the architecture and the integration of U-Net and ResNet: 

In addition to UNet’s architecture, 

1. Residual Block in ResNet:
●	ResNet addresses training challenges in deep neural networks using residual blocks.
●	Employs skip connections in addition to convolution layers.
●	Learns the difference between input and desired output: 
2. Combining U-Net and ResNet (ResUNet):
●	U-Net with ResNet encoder serves as the segmentation model.
●	ResNet acts as the backbone for feature extraction, while U-Net serves as the decoder for segmentation.
●	Inherits U-Net's ability to capture fine feature details and ResNet's capability to learn deep representations.
SR Loss at Hough Space:
1.	Hough Space Transformation:
○	Applies Hough transformation to convert linear representation to polar coordinate format.
○	Represents lines in the Cartesian coordinate system as points in Hough space.
2.	Contrail Segmentation Enhancement:
○	Discretizes Hough space, associating points with possible lines in the image pixel space.
○	Selects lines close to masked pixels, constructing a two-dimensional point feature set in Hough space.
3.	SR Loss Function:
○	Minimizes differences in Hough space between predicted contrail formations and the target.
○	Specifically considers the linear shape of contrails for improved segmentation.
This combined U-Net and ResNet architecture, along with the integration of Dice Loss and SR Loss at Hough space, forms the basis of our second segmentation model. The approach aims to leverage both fine-grained feature details and deep representations to enhance the accuracy of contrail detection.

## Results

![alt_text](https://github.com/rakesh09111996/Contrails-Detection/blob/4d308f52155c73a480a632da2421e159af84fb4b/contrail_results.png)
