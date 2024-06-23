# Image Classification Transformer

![wallpaper.jpg](wallpaper.jpg)

## Problem

#### [Identify the apparels (Fashion MNIST)](https://datahack.analyticsvidhya.com/contest/practice-problem-identify-the-apparels/download/test-file)

> We have total 70,000 images (28 x 28), out of which 60,000 are part of train images with the label of the type of apparel (total classes: 10) and rest 10,000 images are unlabelled (known as test images).The task is to identify the type of apparel for all test images. Given below is the code description for each of the apparel class/label.

|label|description|
|---|---|
|0|T-shirt/top|
|1|Trouser|
|2|Pullover|
|3|Dress|
|4|Coat|
|5|Sandal|
|6|Shirt|
|7|Sneaker|
|8|Bag|
|9|Ankle boot|

> Public and Private split for test images are 40:60.
> The evaluation metric for this challenge is multi-class accuracy.

## Solution

We use a Vision Transformer (ViT) model as a starting point and train a small fully connected network on top of it to classify clothing images. The images from the dataset are preprocessed and augmented with random transformations. This, along with dropout layers, helps the model learn the information in the images effectively. At the end, we evaluate the model's accuracy and confusion matrix to understand its performance. We also create multiple versions of the same model, each aiming to improve on the previous results without replacing them.

## Instructions

* Download the dataset from [DataHack](https://datahack.analyticsvidhya.com/contest/practice-problem-identify-the-apparels/download/test-file)). The expected file directory is:

|Path|Description|
|---|---|
|`data/train.csv`|The CSV file containing the image IDs and the labels of the training dataset.|
|`data/test.csv`|The CSV file containing the image IDs and the labels of the test dataset.|
|`data/train`|The images referrenced by the training dataset.|
|`data/test`|The images referrenced by the test dataset.|

* Run the Jupyter Notebook:

```bash
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

## Results

* TODO FIXME