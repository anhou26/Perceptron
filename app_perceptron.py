import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Forward propogation function from 2026_perceptron.py 
def forward_prop(weight, data, column):
    summation = np.dot(weight, data[:, column]) 
    pred = np.tanh(summation) 
    if pred >= 0:
        return 1 
    else:
        return -1 

# Image preprocessing pipeline for user-uploaded images 
def process_user_image(uploaded_file):
    # 1. Open and convert to grayscale
    img = Image.open(uploaded_file).convert('L')
    img = img.resize((28, 28))
    data = np.asarray(img)
    
    # 2. Check if the image has a white background (average pixel > 127)
    # If it does, invert it so it matches MNIST (black background, white digits)
    if np.mean(data) > 127:
        data = 255 - data
        
    # 3. Flatten to 784 x 1
    return data.reshape(-1, 1)

# Load trained weights 
weights_path = 'trained_weights.npy'
weights = np.load(weights_path)

# Web App Layout
st.title("The Perceptron: 0 vs. 1 Classifier")
st.write("A visual demonstration to the perceptron, the building block of a neural network!")

# Create two tabs for options
tab1, tab2 = st.tabs(["See the training process!", "Upload an image to test the Perceptron!"])

# Tab 1: Training Process & Visualization
with tab1:
    st.header("How the Perceptron Learned")
    st.write("This model trained on **6,000 images** of zeros and ones across 5 epochs, achieving a final test accuracy of over **99.6%** on around 2800 images of unseen data.")
    
    # Display training stats
    st.subheader("Training Epoch Accuracy Log")
    st.text("Epoch 1 | Training Accuracy:  99.50%\n"
            "Epoch 2 | Training Accuracy:  99.85%\n"
            "Epoch 3 | Training Accuracy:  99.87%\n"
            "Epoch 4 | Training Accuracy:  99.87%\n"
            "Epoch 5 | Training Accuracy:  99.93%")
    
    st.divider()
    
    st.subheader("What does the AI 'see'?")
    st.write("This is a visual heatmap of the final, trained **$1 \\times 784$ weight matrix** reshaped back into $28 \\times 28$ pixels. Brighter red areas show pixels the model heavily associates with a zero, while darker blue areas lean toward a one.")
    
    # Reshape weights to 28x28 for visualization
    weight_image = weights.reshape(28, 28)
    
    fig, ax = plt.subplots()
    cax = ax.imshow(weight_image, cmap='coolwarm')
    fig.colorbar(cax)
    ax.axis('off')
    st.pyplot(fig)

with tab2:
    # User Interface
    st.title("Perceptron: 0 vs 1 Classifier")
    st.write("Drag and drop an image of a handwritten 0 or 1 below to test the Perceptron! Note that you can search something like \"handwritten 0 or handwritten 1\" and snip it to drag and drop the image! \n\nPro Tip: Avoid transparent backgrounds and uncentered digits for best results.")

    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image on the website
        st.image(uploaded_file, width=150)
        
        # Process it and make a prediction
        formatted_image = process_user_image(uploaded_file)
        prediction = forward_prop(weights, formatted_image, 0)
        
        # Display the result!
        if prediction == 1:
            st.success("The Perceptron thinks this is a ZERO!")
        else:
            st.info("The Perceptron thinks this is a ONE!")





