<p align="center">
  <img
    src="https://github.com/user-attachments/assets/c40666f4-46ac-4625-8dcb-8693458fd11f"
    alt="INKORA DigitLab — From Pixels to Predictions"
    width="100%"
  />
</p>

<h1 align="center">INKORA DigitLab</h1>

<h3 align="center">
  From Pixels to Predictions — Explore How a CNN Learns to Recognise Handwriting
</h3>

<p align="center">
  An interactive deep-learning laboratory for handwritten digit recognition.
  Understand the mathematics, explore learned features, and watch information
  travel through a real Convolutional Neural Network.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Streamlit-1.67-red?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Deep%20Learning-CNN-purple" alt="Deep Learning">
  <img src="https://img.shields.io/badge/Status-Under%20Development-yellow" alt="Under Development">
</p>

<p align="center">
  <a href="#-project-overview">Overview</a> •
  <a href="#-live-project-experience">Experience</a> •
  <a href="#-understanding-the-cnn">CNN Explained</a> •
  <a href="#-model-architecture">Architecture</a> •
  <a href="#-installation-and-setup">Installation</a> •
  <a href="#-learning-roadmap">Learning Roadmap</a>
</p>

---

## 1. Project Overview

How does a computer recognise a handwritten digit?

To a human, recognising a number is almost effortless. We identify strokes, curves, corners, and spatial patterns without consciously analysing every pixel.

A computer, however, initially receives an image as numerical data.

**INKORA DigitLab** explores how a Convolutional Neural Network (CNN) transforms those numerical values into meaningful visual representations and ultimately predicts one of ten digit classes, from 0 to 9.

This project is designed as an interactive learning environment where users can:

- Experiment with handwritten digit inputs.
- Observe image preprocessing and refinement.
- Explore actual convolution filters and feature maps.
- Understand the role of activation and pooling operations.
- Visualise information moving through a trained CNN.
- Examine prediction probabilities and validation decisions.
- Connect mathematical concepts with their practical implementation.

The objective goes beyond building a digit classifier.

**It is about making the internal behaviour of a neural network easier to observe, question, and understand.**

## 2. The INKORA-DigitLab Experience

### 2.1 Welcome to the Digit Recognition Studio

The application introduces the idea behind INKORA: a handwritten stroke becomes a collection of pixels, pixels form patterns, and learned patterns contribute to a prediction.

<!-- SCREENSHOT 01: Save the landing page screenshot as docs/images/01-home.png -->
<p align="center">
  <img src="https://github.com/user-attachments/assets/f692a390-e35b-4cbf-9e55-45e408b4cbef" alt="INKORA DigitLab landing page" width="90%">
  <br>
  <em>Figure 1. The INKORA DigitLab welcome screen.</em>
</p>

### 2.2 From a Drawing to a Model Input

A handwritten digit is processed into a representation suitable for the CNN.

The model works with grayscale images of dimensions 28 × 28 pixels.


$$
28 \times 28 = 784 \text{ pixel values}
$$

The pixel values are normalised from the range [0, 255] to [0, 1] and reshaped into a four-dimensional input tensor:

$
(1,28,28,1)
$

The dimensions represent:

| Dimension | Meaning |
|---|---|
| 1 | Batch size |
| 28 | Image height |
| 28 | Image width |
| 1 | Grayscale channel |

<!-- SCREENSHOT 02: Save the digit input / drawing interface as docs/images/02-digit-studio.png -->
<p align="center">
  <img src="https://github.com/user-attachments/assets/29fbb6c6-f9f1-452a-9ac0-4de1e4cef687" alt="Handwritten digit input interface" width="90%">
  <br>
  <em>Figure 2. Preparing a handwritten digit for recognition.</em>
</p>

### 2.3 Image Refinement

Raw drawings can vary significantly in size, position, thickness, and background.

INKORA includes an image-refinement stage that prepares the drawing for the model. The preprocessing pipeline includes operations such as cropping the relevant foreground, centring the digit, resizing, normalisation, and reshaping.

The purpose is to create a consistent input representation while preserving the important visual characteristics of the handwriting.

<!-- SCREENSHOT 03: Save the preprocessing/refinement screen as docs/images/03-image-refinement.png -->
<p align="center">
  <img src="https://github.com/user-attachments/assets/1d2c45e6-d6b8-44ef-b593-7b881ce1dea3" alt="Digit Image Working: refinement | preprocessing | Prediction" width="90%">
  <br>
  <em>Figure 3. Refining a handwritten image before inference.</em>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/62fa019b-752e-4655-9245-e769affb7b9f" alt="Digit image refinement and preprocessing" width="90%">
  <br>
  <em>Figure 3.1 . Refining a handwritten image before inference.</em>
</p>

### 2.4 The Model's Response

Once the image is processed, the trained CNN generates a probability distribution over ten possible classes.

INKORA displays:

- The intended digit supplied for the input.
- The digit predicted by the CNN.
- The model's confidence for its predicted class.
- The probability distribution across all ten classes.
- The automated validation agent's decision.

The application also distinguishes between a classification prediction and an acceptance decision.

For example, a model may predict a digit with high confidence but still fail validation if the prediction does not match the intended label.

<!-- SCREENSHOT 04: Save the prediction screen as docs/images/04-prediction.png -->
<p align="center">
  <img src="https://github.com/user-attachments/assets/d5328c76-6f11-47b9-9019-004f4489f573" alt="CNN prediction and validation results" width="90%">
  <br>
  <em>Figure 4. Prediction probabilities and automated validation.</em>
</p>

---

## 3. Understanding the CNN

This section explains the mathematical foundation of the model implemented in INKORA.

A Convolutional Neural Network is a specialised neural network architecture designed to process grid-like data, particularly images.

Unlike a conventional fully connected network, a CNN can exploit spatial relationships between neighbouring pixels.

Its main building blocks include:

1. Convolution
2. Non-linear activation
3. Pooling
4. Feature extraction
5. Flattening
6. Fully connected layers
7. Softmax classification

### 3.1 Convolution: Learning Visual Features

Convolution is one of the fundamental operations in a CNN.

A convolutional layer applies small learnable matrices, called filters or kernels, across an input image.

For a simplified two-dimensional convolution:

$$
Z(i,j)=\sum_u\sum_v X(i+u,j+v)K(u,v)+b
$$

Where:

- $X$ represents the input image or feature map.
- $K$ represents the learnable kernel.
- $b$ represents the bias.
- $Z(i,j)$ is the resulting value at a particular location.

The kernel moves across the input, performing element-wise multiplication and summation at each position.

The result is a feature map.

**Why is this useful?**

Instead of treating every pixel as an isolated value, convolution allows the network to learn local patterns and reuse the same weights across different image locations.

This is known as **weight sharing**.

In INKORA, the first convolutional layer contains 32 filters of size 3 × 3.

With same padding, its output dimensions remain:

$$
28 \times 28 \times 32
$$

The network therefore produces 32 feature-map channels.

### 3.2 ReLU: Introducing Non-linearity

After convolution, the model applies the Rectified Linear Unit activation function.

$$
f(x)=\max(0,x)
$$

ReLU behaves as follows:

- Positive values are retained.
- Negative values are replaced with zero.

Why is this important?

Without non-linear activation functions, stacking multiple linear transformations would still be mathematically equivalent to a single linear transformation.

ReLU allows the network to learn more complex, non-linear relationships in the input data.

### 3.3 Max Pooling: Reducing Spatial Dimensions

Pooling reduces the spatial dimensions of feature maps.

INKORA uses 2 × 2 max pooling.

For every local 2 × 2 region, the operation retains the maximum activation.

For example:

$$
\begin{bmatrix}
1 & 3\\
2 & 4
\end{bmatrix}
\longrightarrow 4
$$

The first pooling operation transforms:

$$
28 \times 28 \times 32
$$

into:

$$
14 \times 14 \times 32
$$

This reduces computation while retaining strong local responses.

### 3.4 Deeper Convolution: Building Richer Representations

The second convolutional layer receives the output of the first pooling operation.

It contains 64 filters of size 3 × 3, followed by ReLU.

Its output is:

$$
14 \times 14 \times 64
$$

The second max-pooling operation then produces:

$$
7 \times 7 \times 64
$$

As information moves through successive layers, the network learns increasingly transformed representations of the input.

However, filters should not be assumed to have guaranteed human-readable meanings. Their actual responses depend on the learned weights and the input image.

### 3.5 Flattening: From Feature Maps to a Vector

The pooled output has three dimensions:

$$
7 \times 7 \times 64
$$

The Flatten operation converts it into a one-dimensional vector:

$$
7 \times 7 \times 64 = 3136
$$

These 3,136 values are passed to the dense classification layers.

### 3.6 Dense Layer: Combining Extracted Features

A fully connected or dense layer combines the extracted features through learned weights.

Its mathematical representation is:

$$
a=f(Wx+b)
$$

Where:

- $x$ is the input vector.
- $W$ is the weight matrix.
- $b$ is the bias vector.
- $f$ is the activation function.

INKORA uses a dense layer containing 128 neurons with ReLU activation.

### 3.7 Dropout: Regularisation

The architecture includes a Dropout layer with a rate of 0.25.

During training, dropout randomly disables a fraction of activations.

This discourages excessive dependence on particular neurons and can help reduce overfitting.

During inference, dropout is inactive.

### 3.8 Softmax: Producing the Final Prediction

The final layer contains ten output neurons, one for each digit class.

Softmax converts the output scores, known as logits, into a normalised probability distribution.

\[
P(y=i)=
\frac{e^{z_i}}
{\sum_{j=0}^{9}e^{z_j}}
\]

Where \(z_i\) represents the logit associated with class \(i\).

The predicted digit is the class with the highest probability:

$$
\hat{y}=\arg\max_i P(y=i)
$$

The ten probabilities sum to one.

---

## 4. Model Architecture

The current INKORA CNN follows this architecture:

| Stage | Operation | Output Shape |
|---|---|---|
| Input | Grayscale image | 28 × 28 × 1 |
| Convolution 1 | 32 filters, 3 × 3, ReLU, same padding | 28 × 28 × 32 |
| Pooling 1 | MaxPool 2 × 2 | 14 × 14 × 32 |
| Convolution 2 | 64 filters, 3 × 3, ReLU, same padding | 14 × 14 × 64 |
| Pooling 2 | MaxPool 2 × 2 | 7 × 7 × 64 |
| Flatten | Convert feature maps to vector | 3,136 |
| Dense | 128 neurons, ReLU | 128 |
| Dropout | Rate 0.25 | 128 |
| Output | 10 neurons, Softmax | 10 |

### Architecture flow

```text
Input Image
  28 × 28 × 1
       |
       v
Conv2D
  32 filters, 3 × 3
       |
      ReLU
       |
       v
MaxPooling
  2 × 2
       |
       v
  14 × 14 × 32
       |
       v
Conv2D
  64 filters, 3 × 3
       |
      ReLU
       |
       v
MaxPooling
  2 × 2
       |
       v
   7 × 7 × 64
       |
       v
    Flatten
     3,136
       |
       v
 Dense (128)
      ReLU
       |
       v
 Dropout (0.25)
       |
       v
 Dense (10)
     Softmax
       |
       v
 Predicted Digit
```

---

## 5. The Interactive Forward-Pass Visualisation

### See the Computation, Not Just the Prediction

One of the central educational features of INKORA is its interactive forward-pass visualisation.

Rather than presenting only a static architecture diagram, the application allows users to observe how a particular input moves through the trained CNN.

<!-- SCREENSHOT 05: Save the complete forward-pass visualisation as docs/images/05-forward-pass.png -->
<p align="center">
  <img src="docs/images/05-forward-pass.png" alt="INKORA interactive CNN forward pass visualisation" width="100%">
  <br>
  <em>Figure 5. Interactive visualisation of the trained CNN's forward pass.</em>
</p>

### What does the visualisation show?

**Stage 1 — Input Pixels**

The original refined image is represented as a 28 × 28 grayscale input tensor.

**Stage 2 — First Convolution**

Selected channels from the first convolutional layer display the responses produced by the learned filters.

**Stage 3 — First Pooling**

The pooled representation illustrates the reduced spatial dimensions.

**Stage 4 — Second Convolution**

The second convolutional layer generates a deeper feature representation using 64 learned filters.

**Stage 5 — Second Pooling**

The resulting representation has dimensions of 7 × 7 × 64.

**Stage 6 — Dense Layer**

The visualisation displays 18 representative neurons from the 128-neuron dense layer.

**Stage 7 — Output Probabilities**

The ten output classes are displayed with their corresponding probabilities and the predicted class.

### Interactive controls

- **Play:** Start the animation.
- **Pause:** Stop at a particular stage.
- **Step Forward:** Move through the computation one step at a time.
- **Replay:** Restart the animation.
- **Speed Control:** Adjust the animation speed.

### A note on numerical accuracy

The visualisation is driven by the refined input, intermediate activations, learned dense-layer weights, and class probabilities from the trained model.

For readability, it displays selected convolution channels and representative dense neurons rather than every internal value.

The full model still performs its computation using all 32 first-layer filters, all 64 second-layer filters, and all 128 dense neurons.

The visualisation is intended to explain the forward computation, not to imply that the model is updating its weights while a user watches the animation.

<!-- OPTIONAL GIF: Save an actual animation recording as docs/images/05-forward-pass.gif -->
<!-- To use a GIF instead of the still image above, replace the image path with:
<img src="docs/images/05-forward-pass.gif" alt="Animated CNN forward pass" width="100%">
-->

---

## 6. Forward Propagation vs Backpropagation

These two processes are related, but they serve different purposes.

### Forward Propagation — Inference

Forward propagation calculates the network's output from an input image.

```text
Image
  ↓
Convolution
  ↓
ReLU
  ↓
Pooling
  ↓
Feature Extraction
  ↓
Dense Layers
  ↓
Softmax
  ↓
Prediction
```

During inference, the trained weights are used to calculate the prediction.

### Backpropagation — Learning

During training, the model compares its prediction with the correct label.

A common classification loss is categorical cross-entropy:

$$
L=-\sum_{i=0}^{9}y_i\log(\hat{y}_i)
$$

For a one-hot encoded target, this simplifies to:

$$
L=-\log(\hat{p}_{correct})
$$

Backpropagation calculates the gradients of the loss with respect to the model's parameters.

An optimiser uses those gradients to update the weights.

A simplified gradient-descent update is:

$$
W_{new}=W_{old}-\eta\frac{\partial L}{\partial W}
$$

Here, $\eta$ represents the learning rate.

INKORA uses the Adam optimiser in its training workflow.

**Important:** The interactive animation demonstrates forward propagation using an already-trained model. It does not demonstrate backpropagation or live weight updates.

---

## 7. Automated Validation Agent

Digit classification and submission validation are two separate responsibilities.

The CNN predicts a digit and provides a probability distribution.

The validation agent applies additional checks to decide whether a submission should be accepted or rejected.

The application includes checks such as:

- Whether the drawing contains sufficient ink.
- Whether the input is excessively clipped at the boundaries.
- Whether the submission duplicates an existing input within the relevant submission context.
- Whether the intended label agrees with the predicted digit.
- Whether the prediction meets the configured confidence threshold.

The distinction is important:

**A high-confidence prediction is not automatically an accepted submission.**

A prediction may be correct but fail another validation condition, or it may be confidently incorrect.

The agent is an additional quality-control layer; it does not replace the CNN or independently establish the model's overall accuracy.

---

## 8. Training and Model Development

The current model is a trained CNN intended for handwritten-digit recognition.

The project also includes a training script:

```text
scripts/train_model.py
```

The model is stored at:

```text
models/mnist_digit_cnn.keras
```

The training workflow is intended to make the model architecture and its learned parameters reproducible and open to further experimentation.

For meaningful model evaluation, training and testing must be treated separately.

A model should not be judged only by how confidently it predicts a few demonstration images.

A robust evaluation process should include:

| Metric | What it tells us |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| Precision | How often predictions for a class are correct |
| Recall | How many actual examples of a class are identified |
| F1-score | Balance between precision and recall |
| Confusion Matrix | Which digit classes are commonly confused |
| Inference Time | How long the model takes to produce predictions |

### Current development status

<em>INKORA DigitLab is an actively evolving project. The current application and CNN inference workflow are functional, but model performance is still under development.</em>

Further work is required to systematically evaluate and improve predictive accuracy, robustness, and generalisation.

No unverified accuracy figure is claimed in this README.

The focus of the next development phase is to establish reliable evaluation baselines and improve performance through controlled experiments.

---

## 9. Improving CNN Performance: Future Experiments

The following are planned directions for further development. They are experiments to investigate, not improvements already established.

### 9.1 Data Augmentation

Handwritten digits can vary in position, scale, rotation, and stroke thickness.

Carefully controlled transformations may help the network generalise to handwriting it has not encountered before.

Potential experiments:

- Small rotations.
- Horizontal and vertical translations.
- Controlled scaling.
- Minor changes in stroke thickness.

Augmentation must preserve the identity of the original digit. Excessive transformations can create misleading training examples.

### 9.2 Batch Normalisation

Batch Normalisation normalises intermediate activations during training and can improve optimisation behaviour.

It is worth comparing architectures with and without batch normalisation.

### 9.3 Learning-Rate Optimisation

The learning rate affects how quickly and effectively a model updates its parameters.

Potential experiments include:

- Learning-rate scheduling.
- Learning-rate reduction on validation plateaus.
- Comparing different initial learning rates.

### 9.4 Regularisation

Overfitting occurs when a model learns the training data too specifically and performs poorly on unseen examples.

Possible techniques include:

- Dropout.
- Weight regularisation.
- Early stopping.
- Data augmentation.

### 9.5 Architecture Experiments

The current two-convolution architecture provides a baseline.

Further experiments could investigate:

- Additional convolutional blocks.
- Different filter sizes.
- Different numbers of filters.
- Alternative pooling strategies.
- Batch normalisation.
- Different dense-layer sizes.

### 9.6 Better Generalisation to Real Handwriting

MNIST-style data and user-generated drawings are not necessarily identical.

The model should be tested on diverse handwriting examples, particularly inputs that vary in stroke thickness, positioning, and writing style.

The ultimate goal is not merely to perform well on familiar benchmark images, but to build a more reliable digit-recognition system.

---

## 10. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | CNN development, training, and inference |
| Streamlit | Interactive application interface |
| NumPy | Numerical operations and tensor handling |
| Pillow | Image processing |
| Git | Version control |
| GitHub | Source-code hosting and project documentation |

---

## 11. Project Structure

```text
INKORA-DigitLab/
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
├── .gitignore
│
├── .streamlit/
│   └── config.toml
│
├── models/
│   └── mnist_digit_cnn.keras
│
├── modules/
│   ├── __init__.py
│   ├── agent.py
│   ├── classifier.py
│   ├── refinery.py
│   └── visualizer.py
│
├── scripts/
│   └── train_model.py
│
├── data/
│   ├── accepted/
│   ├── rejected/
│   └── review_logs/

```

**Module responsibilities**

| File | Responsibility |
|---|---|
| `app.py` | Main Streamlit application and user flow |
| `agent.py` | Automated validation and acceptance logic |
| `classifier.py` | CNN loading and prediction operations |
| `refinery.py` | Image preprocessing and refinement |
| `visualizer.py` | Interactive forward-pass visualisation |
| `train_model.py` | Model training workflow |
| `mnist_digit_cnn.keras` | Saved trained CNN |
| `config.toml` | Streamlit appearance and configuration |

The `data/` directory may contain local application data. User-specific records and databases are not intended to be committed to the public repository.

---

## 12. Installation and Setup

Want to run INKORA locally?

Follow these steps.

### Prerequisites

- Python installed on your system.
- Git.
- A code editor such as Visual Studio Code.
- An internet connection for installing dependencies.

### Step 1 — Clone the repository

```bash
git clone https://github.com/pallavikumariryz1/inkora-digitlab.git
```

### Step 2 — Navigate to the project

```bash
cd inkora-digitlab
```

### Step 3 — Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### Step 4 — Install dependencies

```bash
python -m pip install --upgrade pip
```

```bash
python -m pip install -r requirements.txt
```

### Step 5 — Launch the application

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open the URL in your browser to explore INKORA.

---

## 13. Learning Roadmap: From Beginner to Advanced

INKORA is also intended to be a learning resource.

You can use the project to follow the progression below.

| Level | Concepts to explore | What to understand |
|---|---|---|
| Beginner | Images and pixels | Grayscale, resolution, pixel intensity |
| Beginner | Preprocessing | Cropping, centring, resizing, normalisation |
| Intermediate | Neural networks | Neurons, weights, biases, activation functions |
| Intermediate | Convolution | Kernels, strides, padding, feature maps |
| Intermediate | CNN architecture | ReLU, pooling, flattening, dense layers |
| Advanced | Training | Loss functions, backpropagation, optimisers |
| Advanced | Evaluation | Precision, recall, F1-score, confusion matrix |
| Advanced | Generalisation | Overfitting, regularisation, augmentation |
| Advanced | Interpretability | Intermediate activations and learned representations |

### Suggested experiments

If you want to learn by modifying the project:

1. Change the number of convolution filters and compare the output shapes.
2. Experiment with different kernel sizes.
3. Observe the effect of max pooling.
4. Compare ReLU with another activation function.
5. Investigate how the learning rate affects training.
6. Compare training and validation performance.
7. Examine which digit pairs are frequently confused.
8. Test the model on handwritten inputs that differ from the training distribution.

The best way to understand a neural network is to experiment, measure, and question the results.

---

## 14. Limitations and Responsible Use

A few limitations are important to understand:

- A prediction confidence score is not equivalent to overall model accuracy.
- Performance on MNIST-style images does not guarantee performance on every real-world handwriting style.
- The forward-pass visualisation shows selected internal representations for readability rather than every individual computation.
- The automated validation agent applies application-specific rules and should not be interpreted as a guarantee of classification correctness.
- Model performance and robustness remain under active development.

The project is an educational and experimental digit-recognition application, not a safety-critical recognition system.

---

## 15. Roadmap

- [x] Build an interactive Streamlit application.
- [x] Implement a CNN-based handwritten digit classifier.
- [x] Develop an image refinement pipeline.
- [x] Add automated submission validation.
- [x] Visualise learned convolution filters and feature maps.
- [x] Build an interactive forward-pass visualisation.
- [x] Display prediction probabilities and validation decisions.
- [ ] Establish comprehensive evaluation metrics.
- [ ] Improve model accuracy and generalisation.
- [ ] Experiment with data augmentation.
- [ ] Compare alternative CNN architectures.
- [ ] Expand the educational explanations and visualisations.
- [ ] Evaluate on a broader collection of real handwritten inputs.

---

## 16. Contributing

Ideas, suggestions, and constructive feedback are welcome.

If you are interested in improving INKORA:

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Test your changes locally.
5. Submit a pull request explaining the improvement.

For substantial changes, please describe the motivation and include evaluation results where applicable.

---

## 17. Acknowledgements and Learning Resources

This project builds on foundational ideas in deep learning and computer vision.

Useful resources for exploring the concepts further:

- [TensorFlow Documentation](https://www.tensorflow.org/tutorials/images/cnn)
- [Keras Documentation](https://keras.io/)
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [3Blue1Brown — Neural Networks](https://www.3blue1brown.com/topics/neural-networks)
- [DeepLearning.AI](https://www.deeplearning.ai/)

These resources offer complementary perspectives on the mathematical foundations and practical implementation of neural networks.

---

## 18. Final Note

INKORA DigitLab began with a simple question:

**What actually happens between showing a handwritten digit to a computer and receiving a prediction?**

This project is an attempt to explore that question through code, mathematics, experimentation, and interactive visualisation.

It is still evolving, and improving its performance is an important part of the journey.

The aim is to make deep learning less of a black box and more of something we can explore, understand, and build upon.

<p align="center">
  <strong>INKORA DigitLab</strong>
  <br>
  <em>Don't just see what the model predicts. Explore how it gets there.</em>
</p>

<p align="center">
  Built with curiosity, experimentation, and a love for understanding intelligent systems.
</p>