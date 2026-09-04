# Perceptron
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://0-vs-1-classifier.streamlit.app/)
A from-scratch Python implementation of a single-layer perceptron classifying handwritten 0s and 1s. Built entirely with NumPy to demonstrate foundational ML math—including forward propagation, custom backpropagation, and memory optimization. Features an interactive Streamlit web app for live inference and weight matrix visualization.

Trained on a subset of the MNIST dataset (8,816 images of 0s and 1s), this repository includes an interactive Streamlit web application to visualize the trained weight matrix and test the model with live user uploads.

*(Note: The repository relies on pre-trained weights to keep the size lightweight. The raw training dataset is not included in this repo, but is the MNIST dataset of JPG images from Kaggle by Stuart Colianni https://www.kaggle.com/datasets/scolianni/mnistasjpg?resource=download).*


## Features

* **From-Scratch Mathematics:** Implements matrix multiplication and the $y = \tanh(W \cdot X)$ activation function using only NumPy.
* **Custom Backpropagation:** Updates a $1 \times 784$ weight matrix over 5 epochs using a custom learning rate ($\alpha = 0.002$).
* **High Accuracy:** Achieved **99.64% validation accuracy** on ~2,800 unseen images.
* **Interactive Web App:** A Streamlit interface that allows users to upload custom images for inference and visualizes the trained weight matrix as a 2D heatmap.

## Optimizations & Lessons Learned
This project went through several iterations to improve performance and mathematical accuracy (visible in the commit history):
* **Memory Optimization:** Initial versions used `np.c_` to append columns inside a loop, which forces NumPy to reallocate the entire matrix in memory for every single image ($O(N^2)$). This was optimized by appending arrays to a standard Python list first, and then using `np.hstack()` at the end, drastically reducing data processing time.
* **Activation Function Refinement:** The initial build used a sigmoid function, but continuous decimal probabilities didn't cleanly map to discrete `1` and `-1` classes, causing backpropagation to misfire. Switching to the $\tanh$ function naturally bounded outputs between -1 and 1, creating a much cleaner threshold for classification.

## Project Structure

* `2026_perceptron.py`: The core training script handles raw image preprocessing, dataset compilation/caching (saving to `.npz`), training loops, and saving the final trained weights (`.npy`).
* `app_perceptron.py`: The Streamlit web application loads the pre-trained weights, processes user-uploaded images, and outputs real-time predictions.

## Visualizing the Learning
Instead of a "black box," the Streamlit app reshapes the final $1 \times 784$ weight matrix back into a $28 \times 28$ image. This generates a heatmap showing exactly what the AI "looks for" when classifying an image. Brighter red pixels indicate areas the model strongly associates with a `0`, while darker blue pixels indicate a `1`.

*(Here are some expected inputs: When drawn centrally and with standard proportions, the ink cleanly activates the correct regions of the weight matrix (the blue center for '1', the red outer ring for '0'). The model easily powers through minor noise, like the slash in the zero or a small base on the one. ).*

| Perfect '1' | Thick '1' | Standard '0' | Slashed '0' |
| :---: | :---: | :---: | :---: |
| <img src="Images%20for%20Github/Right.png" width="150"/> | <img src="Images%20for%20Github/Right_1.png" width="150"/> | <img src="Images%20for%20Github/Right_0.png" width="150"/> | <img src="Images%20for%20Github/0_is_right.png" width="150"/> |

## Model Limitations: A Study in Spatial Exactness

Because this is a single-layer perceptron computing a single dot product, it acts as a **spatial template matcher**. It lacks the translation and scale invariance found in Convolutional Neural Networks (CNNs). 

By reshaping the trained $1 \times 784$ weight matrix into a 2D heatmap, we can visualize exactly how the model "thinks":
* **Red Pixels (Positive Weights):** Ink here pushes the prediction toward **ZERO**. The model expects a circular shape.
* **Blue Pixels (Negative Weights):** Ink here pushes the prediction toward **ONE**. The model expects a dense vertical line in the center.

### Successful Predictions
When the input aligns with the spatial structure of the MNIST training data, the perceptron classifies it with high confidence, even with slight variations:

| Perfect '1' | Standard '0' | Noisy '0' (Slashed) |
| :---: | :---: | :---: |
| <img src="Images%20for%20Github/Right.png" width="150"/> | <img src="Images%20for%20Github/Right_0.png" width="150"/> | <img src="Images%20for%20Github/0_is_right.png" width="150"/> |
*Note: In the slashed '0', the strong activation of the red outer ring overpowers the noise of the slash hitting the blue center.*

### Edge Cases & Misclassifications
The model fails when a user draws a digit that spatially triggers the opposing weights. 

| The "Angled" 1 | The "Hooked" 1 |
| :---: | :---: |
| <img src="Images%20for%20Github/Wrong_1.png" width="150"/> | <img src="Images%20for%20Github/Wrong.png" width="150"/> |

* **The Angled 1:** The wide base and slanted stem miss the blue center entirely and activate the bottom-left of the red "ZERO" ring.
* **The Hooked 1:** The large top loop mimics the top arch of a '0', heavily activating the positive red weights and tricking the model into predicting a ZERO.

**Takeaway:** This illustrates the limitations of standard artificial neural networks (ANNs) in computer vision. To solve this edge case and recognize digits regardless of where they are drawn on the canvas, a model requires convolution operations (filters/kernels) to detect localized features rather than global spatial templates.

## License
Distributed under the MIT License.
