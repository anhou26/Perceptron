# Perceptron
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-custom-app-link.streamlit.app)
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

## License
Distributed under the MIT License.
