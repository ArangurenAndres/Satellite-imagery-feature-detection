# U-net based for feature detection of satellite images 

Describe and analyze a model developed for the DSTL satellite imagery dataset from Kaggle competition. We will explore supervised deeplearning models using fully connected neural networks forthe detection of objects labeled in 10 different classes

The dataset is composed of images captured using multispectral sensors which capture the information across several wide-bands (3 to 10) in the electromagnetic spectrum. This multispectral data allows to detect and classify objects based on its reflectance properties across several bands. The dataset consists of 25 labelled images captured across 17 different bands: A band(8 channels) + M band (8 channels) + P band (1 channel)) varying the number of channels and resolution.
An important remark is the fact that there is highly imbalance in the training dataset, in terms of the area in pixels covered by each of the objects labelled in the input image, additionally in some images there is overlapping between classes.

An important remark is the fact that there is highly im-balance in the training dataset, in terms of the area in pixelscovered by each of the objects labelled in the input image,additionally in some images there is overlapping between classes.

<img src="https://github.com/ArangurenAndres/Satellite-imagery-feature-detection/blob/main/Barplots(1).png" width="600" height="500">



<img src="https://github.com/ArangurenAndres/Satellite-imagery-feature-detection/blob/main/Mask%20and%20Bands%202(2).png" width="600" height="600">

## Preprocessing

In order to work with the data in an easier manner, wemerged the 25 training images of 835X835 resolution intoone large scale image that contains the data with good res-olution and the most number of channels as possible, thusproducing a multispectral image of 4175X4175 pixels. Suc-cessively we apply a normalization process which dependson the class we are interested in identifying.  This is due tothe reflectance variation of objects when captured by satel-lites


## Methods

## Validation and training data

The  first  part  of  the  methodology  consists  of  the  splitof  data  sets  into  training  and  validation  sets.   In  order  toprovide an adequate input image resolution to the networkand to reduce computational cost,  we fix an input dimen-sion of the images for both training and validation sets at asquare 160 X 160 pixels dimension.  Following we apply arandom selection process that extracts images of 160X160 pixels of the normalized large scale image, each image hasalso to satisfy a prespecified threshold in order to maximizethe presence of each class considered in the problem.  Fol-lowing an example of the threshold applied for the selectionof images, which consists of an array of 10 elements cor-responding to the classes, where each value determines theminimum percentage of a class presence or area covered inthe considered image.


## U-net architecture

The model that was implemented is known as U-net which is a fully convolutionalNetwork  for  semantic  segmentation. In this case the initial input dimension is (160,160,8) andthe  expected  output  (160,160,1)  since  we  chose  a  binaryclassification approach instead of classifying all the classes in  the  same  order,  due  to  computational  resources  andexecution time limitations. 

## Experiments

## Segmentation of structures

<img src="https://github.com/ArangurenAndres/Satellite-imagery-feature-detection/blob/main/Prediction%20Model%202.png" width="700" height="300">

<img src="https://github.com/ArangurenAndres/Satellite-imagery-feature-detection/blob/main/Prediction%20Model%202_1.png" width="700" height="300">


## Segmentation of standing water
<img src="https://github.com/ArangurenAndres/Satellite-imagery-feature-detection/blob/main/Prediction%20Model%202_1.png" width="700" height="300">

