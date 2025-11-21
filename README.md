# Face-mask-detection-using-CNN-and-OpenCV
Train a CNN to detect whether a person is wearing a mask and run real-time detection using OpenCV + Haarcascade.

# What the code does:
1. Imports required libraries:
   TensorFlow/Keras, NumPy, Matplotlib, OpenCV, Zipfile, ImageDataGenerator.

# 2. Loads dataset:
   - Points to a ZIP file containing two folders:
        /with_mask
        /without_mask
   - Extracts the ZIP to a target directory.
   - Uses ImageDataGenerator with validation_split=0.2 (80% train / 20% validation).

# 3. Creates data loaders:
   train_generator:
       target_size = (64, 64)
       batch_size  = 32
       class_mode  = binary
       subset      = training

   validation_generator:
       same settings, subset = validation

# 4. Visualizes sample images from both training and validation sets.

# 5. Builds a CNN model:
   - Conv2D(64) + MaxPooling2D
   - Conv2D(64) + MaxPooling2D
   - Flatten
   - Dense(128, relu)
   - Dense(1, sigmoid)
   Input shape: (64, 64, 3)

# 6. Compiles and trains:
   optimizer = adam
   loss = binary_crossentropy
   metrics = accuracy
   epochs = 10
   validation_data = validation_generator

# 7. Plots training vs validation accuracy and loss.

# 8. Saves the trained model:
   mask_detection_model.keras

# 9. Loads the saved model for inference.

# 10. Uses OpenCV Haarcascade for face detection:
    haarcascade_frontalface_default.xml

# 11. Real-time webcam detection loop:
    - Detect face in each frame
    - Crop face region
    - Resize to (64, 64)
    - Normalize /255
    - Predict mask vs no mask
    - Draw bounding box
         green  = mask
         red    = no mask
    - Display confidence on screen
    - Press 'q' to exit

# Dependencies:
pip install tensorflow keras numpy matplotlib opencv-python

Run:
1. Place your dataset zip file at the expected location or update the path in the notebook.
2. Open the notebook:
   jupyter notebook
3. Run all cells.
4. After training, run the final cells to start real-time mask detection.

Notes:
- Uses absolute Windows paths; change them to relative paths for portability.
- CNN is simple but effective for binary classification.
- Model expects two directory classes: with_mask / without_mask.
# 12. Results
- 98% accuracy on testing set


