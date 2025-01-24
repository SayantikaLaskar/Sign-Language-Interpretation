import cv2
import numpy as np
from keras.models import load_model

# Define the classes (ASL letters)
classes = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z"
]

# Load the pre-trained model
model = load_model("sign_language_cnn_model.h5")

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Define a function to preprocess the frame
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    resized = cv2.resize(gray, (28, 28))  # Resize to 28x28 to match model input
    normalized = resized / 255.0  # Normalize pixel values
    
    # Reshape for the model (batch_size, height, width, channels)
    reshaped = np.reshape(normalized, (1, 28, 28, 1))
    
    return reshaped


print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a mirror effect
    frame = cv2.flip(frame, 1)

    # Define the region of interest (ROI) for hand detection
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]

    # Preprocess the ROI
    input_data = preprocess_frame(roi)

    # Make predictions
    predictions = model.predict(input_data)
    predicted_label = np.argmax(predictions[0])
    predicted_class = classes[predicted_label]

    # Display the prediction
    cv2.putText(frame, f"Prediction: {predicted_class}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw the ROI on the frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("Sign Language Recognition", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
