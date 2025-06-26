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








'''
while True:
    ret, frame = webcam.read() # returns True/False and one frame

    if ret:
        # Only process game logic if game is not over
        if not game_over:
            current_time = time.time()
            
            # Check if enough time has passed since last round
            if current_time - last_round_time >= round_cooldown:
                hand, accuracy = classify(frame)
                if accuracy > 0.7:
                    rps_hand = hand_options[hand]
                    
                    # Only play a round with valid hand detection
                    comp_hand = computer_choice()
                    round_winner = determine_winner(rps_hand, comp_hand)
                    
                    if round_winner == "Win":
                        player_score += 1
                        show_result_until = current_time + 1.5  # Show result for 1.5 seconds
                    elif round_winner == "Loss":
                        comp_score += 1
                        show_result_until = current_time + 1.5
                    elif round_winner == "Tie":
                        show_result_until = current_time + 1.5
                    
                    # Update last round time and check game over
                    last_round_time = current_time
                    check_game_over()
                else:
                    rps_hand = "Unknown"
                    round_winner = "Invalid"
            else:
                # Still classify hand for display but don't play round
                hand, accuracy = classify(frame)
                if accuracy > 0.7:
                    rps_hand = hand_options[hand]
                else:
                    rps_hand = "Unknown"
                    
                # Show countdown until next round
                time_left = round_cooldown - (current_time - last_round_time)
                cv2.putText(frame, f"Next round in: {time_left:.1f}s", (50, 30), 
                           cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 0), 2)
            
            # Display round results if still showing
            if current_time < show_result_until:
                if round_winner == "Win":
                    cv2.putText(frame, f"You {round_winner}!", (50, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                elif round_winner == "Loss":
                    cv2.putText(frame, f"You {round_winner}..", (50, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                elif round_winner == "Tie":
                    cv2.putText(frame, f"{round_winner}!", (50, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)

        # Display current hand and accuracy (only if game not over)
        if not game_over:
            cv2.putText(frame, f"Hand: {rps_hand}", (10, 100), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Accuracy: {accuracy:.2f}", (10, 130), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show computer's hand only during result display
            current_time = time.time()
            if current_time < show_result_until:
                cv2.putText(frame, f"Computer: {comp_hand}", (10, 160), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"Player: {player_score}", (10, 190), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Computer: {comp_score}", (10, 220), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)

        # Display game over message if game is done
        if game_over:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
            
            text_size = cv2.getTextSize(winner_message, cv2.FONT_HERSHEY_COMPLEX, 1.2, 3)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] + text_size[1]) // 2
            
            cv2.putText(frame, winner_message, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 255, 255), 3)
            cv2.putText(frame, "Press ESC to exit", (text_x + 25, text_y + 50), 
                       cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("RPS model", frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        print("cya")
        break
'''
