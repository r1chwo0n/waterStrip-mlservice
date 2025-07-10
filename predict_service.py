from flask import Flask, request, jsonify
import base64
import numpy as np
from PIL import Image
import io
import cv2
import os
import uuid
from joblib import load

# Load model
model = load("ph_model.joblib")

app = Flask(__name__)

# สร้างโฟลเดอร์ outputs หากยังไม่มี
os.makedirs("outputs", exist_ok=True)

# ----------- ตัดแถบสีจากภาพ -----------
def process_image(image_b64):
    if "," in image_b64:
        image_b64 = image_b64.split(",")[1]
    image_data = base64.b64decode(image_b64)
    image_array = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("ไม่สามารถโหลดภาพได้")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow_tip = np.array([20, 80, 80])
    upper_yellow_tip = np.array([23, 255, 255])
    mask_yellow_tip = cv2.inRange(hsv, lower_yellow_tip, upper_yellow_tip)

    contours, _ = cv2.findContours(mask_yellow_tip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("ไม่พบปลายแถบสีเหลือง")

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    strip_height = 200
    strip = image[y+h:y+h+strip_height, x:x+w]
    strip_hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)

    yellow_tip = image[y:y+h, x:x+w]
    yellow_tip_hsv = cv2.cvtColor(yellow_tip, cv2.COLOR_BGR2HSV)
    h_mean, s_mean, v_mean = cv2.mean(yellow_tip_hsv)[:3]

    lower_dynamic_yellow = np.array([max(0, h_mean - 6), max(0, s_mean - 60), max(0, v_mean - 60)])
    upper_dynamic_yellow = np.array([min(179, h_mean + 6), min(255, s_mean + 60), min(255, v_mean + 60)])

    yellow_mask = cv2.inRange(strip_hsv, lower_dynamic_yellow, upper_dynamic_yellow)

    start_row = 0
    for row in range(strip.shape[0]):
        row_pixels = yellow_mask[row, :]
        yellow_ratio = np.count_nonzero(row_pixels) / row_pixels.size
        if yellow_ratio < 0.9:
            start_row = row
            break

    color_strip = strip[start_row:start_row+100, :]
    color_strip_hsv = cv2.cvtColor(color_strip, cv2.COLOR_BGR2HSV)

    lower_color = np.array([0, 40, 40])
    upper_color = np.array([179, 255, 255])
    color_mask = cv2.inRange(color_strip_hsv, lower_color, upper_color)

    coords = cv2.findNonZero(color_mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        color_cropped = color_strip[y:y+h, x:x+w]
        cropped_path = os.path.join("outputs", f"color_cropped_{uuid.uuid4().hex}.jpg")
        cv2.imwrite(cropped_path, color_cropped)
        return cropped_path
    else:
        raise ValueError("ไม่พบส่วนที่มีสีเพียงพอในการครอป")

# ----------- แปลงภาพเป็นฟีเจอร์ (RGB + HSV) ----------
def extract_features_from_file(image_path):
    pil_image = Image.open(image_path).convert("RGB")
    pil_image = pil_image.resize((50, 50), Image.BILINEAR)
    np_img = np.array(pil_image)

    avg_rgb = np.mean(np_img.reshape(-1, 3), axis=0)
    np_img_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    np_img_hsv = cv2.cvtColor(np_img_bgr, cv2.COLOR_BGR2HSV)
    avg_hsv = np.mean(np_img_hsv.reshape(-1, 3), axis=0)

    features = np.concatenate([avg_rgb, avg_hsv])
    print("Extracted features:", features)
    return features

# ----------- Root -----------
@app.route("/", methods=["GET"])
def index():
    return "Flask ML service is running!", 200

# ----------- /predict -----------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        image_b64 = data.get("image")
        if not image_b64:
            return jsonify({"error": "No image provided"}), 400

        # ตัดแถบสีออกจากภาพ base64
        cropped_path = process_image(image_b64)

        # แปลงภาพครอปเป็นฟีเจอร์
        features = extract_features_from_file(cropped_path)

        # ทำนาย pH
        prediction = model.predict([features])[0]
        final_prediction = max(0, min(14, round(prediction, 2)))

        return jsonify({
            "prediction": final_prediction,
            "cropped_path": cropped_path
        })

    except Exception as e:
        print("Error in /predict:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # fallback for local dev
    app.run(host="0.0.0.0", port=port, debug=True)

