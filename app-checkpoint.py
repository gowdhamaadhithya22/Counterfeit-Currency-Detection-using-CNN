{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": "null",
   "id": "f274359b-c07e-49f6-b6b5-65dedd2cbb0c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import numpy as np\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator \n",
    "from keras.models import Sequential\n",
    "from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense\n",
    "\n",
    "# Define constants\n",
    "IMAGE_WIDTH, IMAGE_HEIGHT = 224, 224\n",
    "BATCH_SIZE = 32\n",
    "EPOCHS = 10\n",
    "TRAIN_DATA_DIR = \"C:\\\\Users\\\\krish\\\\Downloads\\\\archive\\\\Indian Currency Dataset\\\\train\"\n",
    "TEST_DATA_DIR = \"C:\\\\Users\\\\krish\\\\Downloads\\\\archive\\\\Indian Currency Dataset\\\\test\"\n",
    "\n",
    "# Define the CNN model\n",
    "model = Sequential()\n",
    "\n",
    "model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)))\n",
    "model.add(MaxPooling2D(pool_size=(2, 2)))\n",
    "\n",
    "model.add(Conv2D(64, (3, 3), activation='relu'))\n",
    "model.add(MaxPooling2D(pool_size=(2, 2)))\n",
    "\n",
    "model.add(Conv2D(128, (3, 3), activation='relu'))\n",
    "model.add(MaxPooling2D(pool_size=(2, 2)))\n",
    "\n",
    "model.add(Flatten())\n",
    "\n",
    "model.add(Dense(512, activation='relu'))\n",
    "model.add(Dense(1, activation='sigmoid'))  # Binary classification (fake or real)\n",
    "\n",
    "# Compile the model\n",
    "model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])\n",
    "\n",
    "# Data Augmentation\n",
    "train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)\n",
    "test_datagen = ImageDataGenerator(rescale=1./255)\n",
    "\n",
    "train_generator = train_datagen.flow_from_directory(\n",
    "    TRAIN_DATA_DIR,\n",
    "    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),\n",
    "    batch_size=BATCH_SIZE,\n",
    "    class_mode='binary')\n",
    "\n",
    "test_generator = test_datagen.flow_from_directory(\n",
    "    TEST_DATA_DIR,\n",
    "    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),\n",
    "    batch_size=BATCH_SIZE,\n",
    "    class_mode='binary')\n",
    "\n",
    "# Train the model\n",
    "model.fit(\n",
    "    train_generator,\n",
    "    steps_per_epoch=train_generator.samples // BATCH_SIZE,\n",
    "    epochs=EPOCHS,\n",
    "    validation_data=test_generator,\n",
    "    validation_steps=test_generator.samples // BATCH_SIZE)\n",
    "\n",
    "# Save the model\n",
    "model.save('fake_currency_detection_model.h5')\n",
    "\n",
    "# Predict using the model\n",
    "def predict_currency(image_path):\n",
    "    from keras.preprocessing import image\n",
    "    img = image.load_img(image_path, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))\n",
    "    img_array = image.img_to_array(img)\n",
    "    img_array = np.expand_dims(img_array, axis=0)\n",
    "    result = model.predict(img_array)\n",
    "    if result[0][0] > 0.5:\n",
    "        return \"Real Currency\"\n",
    "    else:\n",
    "        return \"Fake Currency\"\n",
    "\n",
    "# Example usage\n",
    "image_path = \"C:\\\\Users\\\\krish\\\\Downloads\\\\archive\\\\Indian Currency Dataset\\\\train\\\\real\\\\1 (44).jpg\"\n",
    "prediction = predict_currency(image_path)\n",
    "print(\"Prediction:\", prediction)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
