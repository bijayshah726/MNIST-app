import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
import cv2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def load_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normalize pixel values to range [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Reshape images to flat vectors
    x_train = x_train.reshape((-1, 28 * 28))
    x_test = x_test.reshape((-1, 28 * 28))

    # Convert labels to one-hot encoded format
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test

def display_digit():

    (x_train_img, y_train), (_,_)= mnist.load_data()
    # Normalize pixel values to range [0, 1]
    x_train = x_train_img.astype('float32') / 255.0

    # Reshape images to flat vectors
    x_train = x_train.reshape((-1, 28 * 28))

    # Display a sample digit and its label
    i = np.random.randint(0, len(x_train)-1)

    return x_train_img[i]



def train_model():
    # Load and preprocess the MNIST dataset
    x_train, y_train, x_test, y_test = load_dataset()

    # Build the CNN model
    model = Sequential()

    # Add a convolutional layer with 32 filters, each of size 3x3
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))

    # Add a max-pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Add another convolutional layer with 64 filters
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

    # Flatten the output before feeding it into the dense layers
    model.add(Flatten())

    # Add a fully connected (dense) layer with 128 units
    model.add(Dense(128, activation='relu'))

    # Add a dropout layer to prevent overfitting
    model.add(Dropout(0.5))

    # Output layer with 10 units (one for each class) and softmax activation
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(x_train.reshape(-1, 28, 28, 1), y_train, batch_size=128, epochs=10, validation_split=0.1)

    # Evaluate the model on the test data
    test_loss, test_acc = model.evaluate(x_test.reshape(-1, 28, 28, 1), y_test)
    print(f"Test accuracy: {test_acc:.4f}")

    model.save("trained_model.h5")


if 0:
    train_model()


# Load your trained model
model = tf.keras.models.load_model('trained_model.h5')

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        # Check if the layer has a 4D output shape (convolutional layer)
        if len(layer.output_shape) == 4:
            return layer.name
    # If no 4D convolutional layer is found, raise an error
    raise ValueError("Could not find 4D convolutional layer.")

def get_output_and_predictions(model, layerName, inputs):
    gradModel = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layerName).output, model.output])

    return gradModel(inputs)


def grad_cam(image, model, classIdx, layerName=None, eps=1e-8, alpha=0.5, colormap='viridis'):
    if layerName is None:
        layerName = find_last_conv_layer(model)

    with tf.GradientTape() as tape:
        inputs = tf.cast(image, tf.float32)
        (convOutputs, predictions) = get_output_and_predictions(model, layerName, inputs)
        loss = predictions[:, classIdx]

    grads = tape.gradient(loss, convOutputs)
    print(grads.shape, "grads")
    castConvOutputs = tf.cast(convOutputs > 0, "float32")
    castGrads = tf.cast(grads > 0, "float32")
    guidedGrads = castConvOutputs * castGrads * grads
    convOutputs = convOutputs[0]
    guidedGrads = guidedGrads[0]

    weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
    print(image.shape, "inside gradcam")
    print(cam.shape)

    (w, h) = (image.shape[2], image.shape[1])
    heatmap = cv2.resize(cam.numpy(), (w, h))  #tf.image resize not working
    numer = heatmap - tf.reduce_min(heatmap)
    denom = (tf.reduce_max(heatmap) - tf.reduce_min(heatmap)) + eps
    heatmap = numer / denom
    print("outside", heatmap.shape)

    # Apply colormap using matplotlib's built-in colormaps
    heatmap_colored = plt.get_cmap(colormap)(heatmap.numpy())[..., :3]
    # Normalize the heatmap_colored to values between 0 and 1
    heatmap_colored = heatmap_colored / np.max(heatmap_colored)
    print(heatmap_colored.shape, image.shape)

    heatmap_overlay = alpha * image[0] + (1 - alpha) * heatmap_colored
    heatmap_overlay = heatmap_overlay / np.max(heatmap_overlay)  # Normalize
    print(heatmap_overlay.shape, image.shape)

    heatmap_overlay = Image.fromarray(np.uint8(heatmap_overlay*255))  #conversion to image format
    heatmap = Image.fromarray(np.uint8(heatmap*255))

    return heatmap_overlay, heatmap

# Streamlit UI
st.title("Mnist Digit Classifier with Grad-CAM")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
st.write("You can also use the 'Display Random Digit' button in the sidebar to visualize a random digit and its prediction.")


# Display Digit Section
st.sidebar.title("Display Random Digit")
if st.sidebar.button("Display Random Digit"):
    random_digit_img = display_digit()
    st.sidebar.image(random_digit_img, caption="Random Digit Image", use_column_width=True)
    
    # Predict using the random digit image
    prediction = model.predict(np.expand_dims(random_digit_img, axis=0))
    predicted_class = np.argmax(prediction)
    st.sidebar.write(f"Predicted Class: {predicted_class}")
    img_array = np.expand_dims(random_digit_img, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    # heat map of MNIST random images if needed
    #heatmap_overlay, heatmap = grad_cam(img_array, model, predicted_class)
    #st.image([heatmap_overlay, heatmap],caption=["Overlayed Image", "Grad-CAM Heatmap"],use_column_width=True)

if uploaded_image is not None:
    try:
        # Check if the file format is valid
        valid_formats = ["png", "jpeg", "jpg"]
        file_extension = uploaded_image.name.split(".")[-1].lower()
        if file_extension not in valid_formats:
            raise ValueError(f"Invalid file format. Supported formats: {', '.join(valid_formats)}")

        pil_image = Image.open(uploaded_image)
        # Convert the image to grayscale mode
        pil_image = pil_image.convert("L")
        img_array = np.array(pil_image)
        print(img_array.shape)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)
        #Reshape the image to 28x28
        img_array = tf.image.resize(img_array, (28, 28))


        st.image(pil_image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            print(img_array.shape, "inside predict")
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            st.write(f"Predicted Class: {predicted_class}")

            heatmap_overlay, heatmap = grad_cam(img_array, model, predicted_class)
            st.image([heatmap_overlay, heatmap],
                     caption=["Overlayed Image", "Grad-CAM Heatmap"],
                     use_column_width=True)
            
    except ValueError as ve:
        st.error(str(ve))
    except Exception as e:
        st.error("An error occurred during image processing.")
