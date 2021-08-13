# ReadyML - Easy and Ready Machine Learning.

## What is ReadyML?

**ReadyML** makes trained Machine Learning models ready to consume with minimum effort. With **ReadyML** you can to play with Image classification, Object Detection, Image Generation, Face Generation, Pose Detection... and much more!

*This library is currently a prototype. If you face any issue or if the documentation is unclear, feel free to open a [ticket issue](https://github.com/houseofai/readyml/issues)*

## Model Categories
| Name | Description |
|-|-|
| [Image Classification](#user-content-image-classification) | Provides category labels for an image |
| [Object Detection](#user-content-objet-detection) | Detect objects in a image |
| [Image Generation](#user-content-image-generation) | Generate image from a category from scratch |
| [Face Generation](#user-content-face-generation) | Generate a face image from scratch |
| [Pose Detection](#user-content-pose-detection) | Detect the keypoints of human pose (ankles, shoulders, elbows, ...) |


## Prerequisites
- [Python](https://www.python.org/downloads/) >= 3.7
- The python package manager: [pip](https://pip.pypa.io/en/stable/installation/)

## Installation
ReadyML installation is very convenient as it is packaged as a pip package. To install it, just run:
```
pip install readyml
```
`pip` will automatically install the dependencies like Tensorflow, Pytorch, Caffe...

## How to use it?
The general format to use a model (`infer`) is that way:
```python
from readyml import <model-category> as ric

# Initialize the model
model = ric.<TheModelIwantToUse>()

# Run the model
results = model.infer(<arguments>)

# Do something with the results
```
The format of the result differs from model to model. See the model names below to get a detailed description.

For example, to use the image classification model `NASNetLarge`, do:
```python
from readyml import imageclassification as ric
import PIL.Image as Image

# Read an image
image_pil = Image.open("./images/greek_street.jpeg")

# Instantiate the model class
nasnetlarge = ric.NASNetLarge()
# Get and print the results
results = nasnetlarge.infer(image_pil)
print(results)
```
**Results:** The labels and their confidence score in percent.
```json
[
    {
        "label": "monastery",
        "score": 38.63
    },
    {
        "label": "palace",
        "score": 21.18
    },
    {
        "label": "patio",
        "score": 14.9
    }
]

```
The above example means that the image can be categorized into three categories:
- `Monastery` with a confidence score of `38.63%`,
- `Palace` with a confidence score of `21.18%`, or
- `Patio` with a confidence score of `14.9%`,

Once ReadyML is intalled, you can start to use all the available models. Check the [Models](models.md) section
