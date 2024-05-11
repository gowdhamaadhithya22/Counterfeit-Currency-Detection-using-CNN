import streamlit as st # type: ignore
from keras.preprocessing import image # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
import numpy as np # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score # type: ignore

# Load the trained model
model = load_model("fake_currency_detection_model1.h5")
IMAGE_WIDTH, IMAGE_HEIGHT = (124, 224)  # Adjust according to your model's input size

# Set theme and background color
st.set_page_config(page_title="Fake Currency Detection", page_icon=":money_with_wings:")
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Placeholder for test generator, replace it with your actual test generator
class TestGenerator:
    def reset(self):
        pass

# Streamlit app
st.title("Counterfeit Currency Detection")


uploaded_file = st.file_uploader("Upload an  Currency image to check if it is fake or real.", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    
    # Predict currency
    img = image.load_img(uploaded_file, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    result = model.predict(img_array)
    prediction = "Real Currency" if result[0][0] > 0.5 else "Fake Currency"
    st.write("Prediction:", prediction)

    # Placeholder for test data, replace it with your actual test data
    # Load test data
    test_path =  "C:\\Users\\krish\\Desktop\\Final Year Project\\archive\\Indian Currency Dataset\\test"
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
                            test_path,
                            target_size=(124, 224),
                            batch_size=32,
                            color_mode="rgb",
                            class_mode="categorical",
                            shuffle=True
                        )
    
    test_generator.reset()
    y_pred = (model.predict(test_generator) > 0.5).astype("int32")
    y_true = test_generator.classes

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Display metrics
    st.write("Accuracy:", accuracy)
    st.write("Precision:", precision)
    st.write("Recall:", recall)
    st.write("F1 Score:", f1)
