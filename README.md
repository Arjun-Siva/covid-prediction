# COVID-19 diagnosis with Deep Learning
![cc](https://user-images.githubusercontent.com/59311154/147398688-76671fcf-ba72-46ab-b5e5-00b31046c623.png)
Diagnosis of COVID-19 from chest X-Ray images with Convolutional Neural Networks and Machine Learning models
## Requirements
- Python 3.7.12
- Tensorflow 2.7.0
- Scikit-learn 1.0.1
- Pandas 1.1.5
- Numpy 1.19.5
- Matplotlib 3.2.2
- Flask 1.1.4

## Architecture
![arch](https://user-images.githubusercontent.com/59311154/147398860-dda20cb7-51b2-4725-9cd3-48074a728e4b.png)
The architecture is largely based on EMCNet[^1]. The images are scaled to (299,299,3) and normalized. The CNN was trained in **Google Colab** with GPU runtime. The CNN is used to extract features from the images. It outputs a vector of dimensions 64x1. This feature vector is used to train the machine learning models and an ensemble of the 4 classifiers - Decision Tree, Support Vector Machine, Random Forest and AdaBoost proved to deliver the maximum accuracy.
![cnn drawio](https://user-images.githubusercontent.com/59311154/147398969-ec2a7644-5928-4300-beed-d7e98a98aa36.png)
The CNN has 20 layers of various types including Conv2D, MaxPooling2D, Dropout and FCL. ReLu activation function is used for the inner layers and dropout threshold of 0.25. The CNN reached an accuracy of **97.10%**. The final Soft voting classifier reached an accuracy of **97.51%**. The intermediate layers' activations are visualized to get a sense of what's going on inside the CNN[^2].

## Dataset
A total of 7232 (3616 images of COVID-19 and 3616 images of Normal) chest X-Rays images were taken from the Kaggle [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database) out of which 5062 were used for training, 1446 for validation and 724 for testing.

## API
The final model was deployed using a Flask API. To run the API,
```
cd api
python app.py
```

[^1]: Saha, P., Sadi, M. S., & Islam, M. M. (2021). EMCNet: Automated COVID-19 diagnosis from X-ray images using convolutional neural network and ensemble of machine learning classifiers. Informatics in medicine unlocked, 22, 100505. [Link](https://www.sciencedirect.com/science/article/pii/S2352914820306560)
[^2]: Companion notebook for the book Deep Learning with Python, Second Edition. [Link](https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/chapter09_part03_interpreting-what-convnets-learn.ipynb)
