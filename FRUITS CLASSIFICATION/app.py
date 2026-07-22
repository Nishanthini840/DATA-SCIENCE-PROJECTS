from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sqlite3
import time
import base64
from datetime import datetime

app = Flask(__name__)

# -------------------------------
# Base directory & Upload folder
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
DB_PATH = os.path.join(BASE_DIR, "scans.db")

# -------------------------------
# Database Setup
# -------------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            fruit TEXT NOT NULL,
            stage TEXT NOT NULL,
            confidence REAL NOT NULL,
            prediction_time REAL NOT NULL,
            timestamp TEXT NOT NULL,
            class_name TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# -------------------------------
# Load trained model
# -------------------------------
MODEL_PATH = os.path.join(BASE_DIR, "fruit_ripeness_classifier_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)

# -------------------------------
# Class labels
# -------------------------------
class_names = [
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges",
    "unripe apple",
    "unripe banana",
    "unripe orange"
]

# -------------------------------
# Helper Functions
# -------------------------------
def process_and_predict(filepath):
    start_time = time.perf_counter()
    
    img = Image.open(filepath).convert("RGB")
    img_resized = img.resize((224, 224))

    img_array = np.array(img_resized, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array, verbose=0)
    index = int(np.argmax(prediction))
    predicted_class = class_names[index]
    confidence = float(prediction[0][index]) * 100
    
    end_time = time.perf_counter()
    pred_duration = round(end_time - start_time, 3)

    # Determine Fruit Name
    if "apple" in predicted_class:
        fruit = "Apple"
    elif "banana" in predicted_class:
        fruit = "Banana"
    elif "orange" in predicted_class:
        fruit = "Orange"
    else:
        fruit = "Unknown"

    # Determine Ripeness Stage & Color
    if "fresh" in predicted_class:
        stage = "Ripe"
        color = "green"
    elif "rotten" in predicted_class:
        stage = "Rotten"
        color = "red"
    else:
        stage = "Unripe"
        color = "orange"

    return {
        "fruit": fruit,
        "stage": stage,
        "confidence": confidence,
        "prediction_time": pred_duration,
        "class_name": predicted_class,
        "color": color
    }

def save_scan_record(filename, result):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO scans (filename, fruit, stage, confidence, prediction_time, timestamp, class_name)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        filename,
        result["fruit"],
        result["stage"],
        result["confidence"],
        result["prediction_time"],
        timestamp,
        result["class_name"]
    ))
    conn.commit()
    scan_id = cursor.lastrowid
    conn.close()
    return scan_id, timestamp

def fetch_scans(limit=100):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM scans ORDER BY id DESC LIMIT ?', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def fetch_stats():
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) as total FROM scans')
    total = cursor.fetchone()['total']
    
    cursor.execute('SELECT AVG(confidence) as avg_conf FROM scans')
    avg_conf_row = cursor.fetchone()['avg_conf']
    avg_conf = round(avg_conf_row, 2) if avg_conf_row else 0.0

    # Fruit breakdown
    cursor.execute('SELECT fruit, COUNT(*) as count FROM scans GROUP BY fruit')
    fruit_counts = {row['fruit']: row['count'] for row in cursor.fetchall()}

    # Stage breakdown
    cursor.execute('SELECT stage, COUNT(*) as count FROM scans GROUP BY stage')
    stage_counts = {row['stage']: row['count'] for row in cursor.fetchall()}
    
    # Recent 10 confidence scores
    cursor.execute('SELECT confidence, timestamp FROM scans ORDER BY id DESC LIMIT 10')
    recent = [dict(row) for row in cursor.fetchall()]
    recent.reverse()

    conn.close()
    return {
        "total_scans": total,
        "avg_confidence": avg_conf,
        "fruit_counts": fruit_counts,
        "stage_counts": stage_counts,
        "recent_confidence": recent
    }

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def home():
    history = fetch_scans(limit=10)
    stats = fetch_stats()
    return render_template("index.html", history=history, stats=stats)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
            return jsonify({"error": "No image uploaded"}), 400
        return render_template("index.html", error="No image provided", history=fetch_scans(10), stats=fetch_stats())

    file = request.files["image"]
    if file.filename == "":
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
            return jsonify({"error": "No file selected"}), 400
        return render_template("index.html", error="No file selected", history=fetch_scans(10), stats=fetch_stats())

    timestamp_prefix = datetime.now().strftime("%Y%m%d_%H%M%S_")
    safe_filename = timestamp_prefix + os.path.basename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], safe_filename)
    file.save(filepath)

    result = process_and_predict(filepath)
    scan_id, timestamp = save_scan_record(safe_filename, result)

    response_data = {
        "id": scan_id,
        "image": safe_filename,
        "fruit": result["fruit"],
        "stage": result["stage"],
        "confidence": f"{result['confidence']:.2f}",
        "raw_confidence": round(result["confidence"], 2),
        "prediction_time": result["prediction_time"],
        "class_name": result["class_name"],
        "color": result["color"],
        "timestamp": timestamp
    }

    if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.accept_mimetypes.accept_json:
        return jsonify(response_data)

    return render_template(
        "index.html",
        image=safe_filename,
        fruit=result["fruit"],
        stage=result["stage"],
        confidence=f"{result['confidence']:.2f}%",
        prediction_time=result["prediction_time"],
        color=result["color"],
        history=fetch_scans(10),
        stats=fetch_stats()
    )

@app.route("/predict-webcam", methods=["POST"])
def predict_webcam():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image data provided"}), 400

    img_data = data["image"]
    if "," in img_data:
        img_data = img_data.split(",")[1]

    try:
        image_bytes = base64.b64decode(img_data)
        filename = f"webcam_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        
        with open(filepath, "wb") as f:
            f.write(image_bytes)

        result = process_and_predict(filepath)
        scan_id, timestamp = save_scan_record(filename, result)

        return jsonify({
            "id": scan_id,
            "image": filename,
            "fruit": result["fruit"],
            "stage": result["stage"],
            "confidence": f"{result['confidence']:.2f}",
            "raw_confidence": round(result["confidence"], 2),
            "prediction_time": result["prediction_time"],
            "class_name": result["class_name"],
            "color": result["color"],
            "timestamp": timestamp
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/history", methods=["GET"])
def api_history():
    scans = fetch_scans(limit=100)
    return jsonify(scans)

@app.route("/api/history/delete/<int:scan_id>", methods=["POST", "DELETE"])
def delete_scan(scan_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT filename FROM scans WHERE id = ?", (scan_id,))
    row = cursor.fetchone()
    if row:
        filename = row["filename"]
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass
        cursor.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
        conn.commit()
        conn.close()
        return jsonify({"success": True, "id": scan_id})
    conn.close()
    return jsonify({"error": "Record not found"}), 404

@app.route("/api/stats", methods=["GET"])
def api_stats():
    return jsonify(fetch_stats())

if __name__ == "__main__":
    app.run(debug=True)