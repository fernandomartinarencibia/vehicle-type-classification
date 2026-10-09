# Vehicle Type Classification: MLP, CNN and Transfer Learning

> Experimental comparison of three neural network approaches for classifying vehicle images into 7 types, from a multilayer perceptron to fine-tuned MobileNetV2, built with TensorFlow/Keras.
> Individual coursework · BSc in Computer Science and Engineering · Universidad Carlos III de Madrid (2025–2026)

**Language:** English · [Español](./README.es.md)

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)](https://www.tensorflow.org/)

---

## Results at a glance

| Model | Input | Validation accuracy |
| :--- | :--- | :---: |
| Majority-class baseline | — | 29.8% |
| MLP | 64×64, grayscale | ≈ 54% |
| Custom CNN, no augmentation | 128×128, RGB | ≈ 86% |
| **Custom CNN + augmentation + class weights** | 128×128, RGB | **92.4%** |
| MobileNetV2 (transfer learning + fine-tuning) | 224×224, RGB | 89.6% |

The key findings: convolutions lift accuracy from about 54% to 86%; data augmentation cuts overfitting and adds about 6 more points; and a small CNN trained from scratch (about 490k parameters) matches or beats a pretrained MobileNetV2 on this dataset.

---

## The data

A Kaggle dataset of **15,645 vehicle images** in 7 imbalanced classes, split 80/20 into training (12,516) and validation (3,129). The dataset is not included in this repository.

| Class | Images |
| :--- | ---: |
| City Car | 4,657 |
| Multi Purpose Vehicle | 2,649 |
| Sedan | 2,619 |
| Van | 2,167 |
| Sport Utility Vehicle | 1,826 |
| Truck | 1,150 |
| Big Truck | 577 |

The largest class has 8 times more images than the smallest, so the loss is weighted by inverse class frequency. Before modelling, `analyze_resolutions.py` analyses the distribution of image resolutions to choose the network input size.

---

## Experiments

**1. MLP baseline.** The image is flattened into a vector, so the first layer's weights grow with the input size. Converting to grayscale cuts the first-layer weights from 1.57M to 524k. The MLP reaches about 54%, well above the 29.8% majority baseline but far from useful: flattening throws away the spatial structure of the image.

**2. Custom CNN.** Four Conv 3×3 + BatchNorm + MaxPool blocks (32 → 256 filters), global average pooling and a dense classifier with dropout. Adam, learning-rate reduction on plateau and early stopping. Without augmentation it overfits badly: 100% training accuracy against about 86% on validation.

**3. Custom CNN with augmentation.** Augmentation layers inside the model (horizontal flip, ±11° rotation, ±10% translation and zoom), plus class weights and stronger dropout. Validation accuracy rises to **92.4%**, and the gap with training shrinks from about 14 to about 3 points.

<table>
<tr>
<td><img src="figures/training_curves_cnn_no_augmentation.png" alt="Training curves without augmentation"></td>
</tr>
<tr><td align="center">Custom CNN without augmentation: training accuracy reaches 100% while validation stalls (overfitting)</td></tr>
<tr>
<td><img src="figures/training_curves_cnn_augmentation.png" alt="Training curves with augmentation"></td>
</tr>
<tr><td align="center">Custom CNN with augmentation: training and validation stay close</td></tr>
</table>

*The loss axis in these plots is mislabelled "binary crossentropy"; the loss is actually sparse categorical cross-entropy.*

**4. Transfer learning with MobileNetV2** (pretrained on ImageNet, 224×224), in two phases: first only the new classifier head is trained with the base frozen (67% validation accuracy), then the upper layers of the base (from layer 100 onwards) are unfrozen and fine-tuned with a very low learning rate, early stopping and checkpointing. Final validation accuracy: **89.6%**.

---

## Limitations

- **No independent test set.** The validation set is also used for early stopping and learning-rate scheduling, so the reported accuracies are somewhat optimistic.
- **Accuracy only.** With imbalanced classes, a confusion matrix and macro-F1 would show how the models handle minority classes such as *Big Truck*.
- **MobileNetV2 used a different split** (seed 123 instead of 42), so its comparison with the CNNs is approximate and the 3-point gap is not conclusive.
- **The notebook was run across several sessions.** Some outputs come from earlier runs; each case is noted next to the relevant cell.

**Next steps:** a fixed train/validation/test split shared by all models, per-model confusion matrices and macro-F1, and a fairer fine-tuning comparison with MobileNetV2 or EfficientNet.

---

## Tech stack

Python, TensorFlow/Keras, scikit-learn, NumPy, pandas, Matplotlib, seaborn.

---

## Repository structure

```
vehicle-type-classification/
├── vehicle_classification.ipynb   # Experiments: MLP, custom CNN, augmentation and MobileNetV2
├── analyze_resolutions.py         # Image resolution analysis used to choose the input size
├── figures/                       # Training curves
├── requirements.txt
├── README.md
└── README.es.md
```

To run it, install the dependencies with `pip install -r requirements.txt` and place the dataset in `data/archive`, with one folder per class. The notebook uses `model.fit(workers=...)`, which requires TensorFlow < 2.16.

---

## Author

**Fernando Martín Arencibia** · [LinkedIn](https://www.linkedin.com/in/fernando-martin-arencibia-477257368/) · [GitHub](https://github.com/fernandomartinarencibia) · [Email](mailto:fernandomartinarencibia@gmail.com)
