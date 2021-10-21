# U-net based for feature detection of satellite images 

Describe and analyze a model developed for the DSTL satellite imagery dataset from Kaggle competition. We will explore supervised deeplearning models using fully connected neural networks forthe detection of objects labeled in 10 different classes

The dataset is composed of images captured using multispectral sensors which capture the information across several wide-bands (3 to 10) in the electromagnetic spectrum. This multispectral data allows to detect and classify objects based on its reflectance properties across several bands. The dataset consists of 25 labelled images captured across 17 different bands: A band(8 channels) + M band (8 channels) + P band (1 channel)) varying the number of channels and resolution.
An important remark is the fact that there is highly imbalance in the training dataset, in terms of the area in pixels covered by each of the objects labelled in the input image, additionally in some images there is overlapping between classes.


