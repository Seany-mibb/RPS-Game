from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import tensorflow as tf
import base64
import io
from PIL import Image
import time
from flask_cors import CORS

app = Flask(__name__)
# CORS allows a web app running on one origin to safely request resources from a
# different origin by letting the server explicitly permit it, preventing security risks
# from unauthorized cross-site requests.
CORS(app)


# Custom function to handle the groups parameter issue on Mac
def custom_depthwise_conv2d(*args, **kwargs):
    # Remove the problematic 'groups' parameter
    if 'groups' in kwargs:
        kwargs.pop('groups')
    return tf.keras.layers.DepthwiseConv2D(*args, **kwargs)

# Load model with custom objects to fix Mac compatibility
try:
    model = tf.keras.models.load_model(
        "keras_model.h5", 
        custom_objects={'DepthwiseConv2D': custom_depthwise_conv2d},
        compile=False
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Trying alternative loading method...")
    try:
        model = tf.keras.models.load_model("keras_model.h5", compile=False)
        print("Model loaded with alternative method!")
    except Exception as e2:
        print(f"Failed to load model: {e2}")
        exit(1)

player_score, comp_score = 0, 0
hand_options = ["Rock", "Paper", "Scissors"]
hand, accuracy = "", ""
game_over = False
winner_message = ""
last_round_time = 0
round_cooldown = 3.0  # 3 seconds between rounds
show_result_until = 0

def computer_choice():
    return np.random.choice(hand_options)

def classify(frame):
    img = cv2.resize(frame, (224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    normalize_array = img_array / 255.0
    prediction = model.predict(normalize_array, verbose=0)  # Added verbose=0 to reduce output
    hand = np.argmax(prediction[0])
    accuracy = np.max(prediction[0])

    return hand, accuracy

def determine_winner(rps_hand, comp_hand):
    if rps_hand == comp_hand:
        return "Tie"
    elif rps_hand == "Rock":
        if comp_hand == "Scissors":
            return "Win"
        else:  # computer_choice == "Paper"
            return "Loss"
    elif rps_hand == "Paper":
        if comp_hand == "Rock":
            return "Win"
        else:  # computer_choice == "Scissors"
            return "Loss"
    elif rps_hand == "Scissors":
        if comp_hand == "Paper":
            return "Win"
        else:  # computer_choice == "Rock"
            return "Loss"
    else:
        return "Invalid"

def check_game_over():
    global game_over, winner_message
    if player_score >= 3:
        game_over = True
        winner_message = "PLAYER WINS THE GAME!"
        return True
    elif comp_score >= 3:
        game_over = True
        winner_message = "COMPUTER WINS THE GAME!"
        return True
    return False

def preprocess_image(image_data):
    # Remove the "data:image/jpeg;base64," prefix if it exists
    if "," in image_data:
        image_data = image_data.split(",")[1]
    
    # Decode Base64 to bytes
    image_bytes = base64.b64decode(image_data)

    # Convert bytes to PIL image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Convert to OpenCV format (BGR for model)
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    return image_cv  # Return the OpenCV image for further processing

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/classify", methods=["POST", "OPTIONS"])
def classify_image():

    data = request.get_json()
    image_data = data.get('image')

    if not image_data:
        return jsonify({'error': 'No image data received'}), 400

    try:
        frame = preprocess_image(image_data)
        hand_idx, acc = classify(frame)
        hand_name = hand_options[hand_idx]

        return jsonify({
            'hand': hand_name,
            'accuracy': round(float(acc), 3)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
