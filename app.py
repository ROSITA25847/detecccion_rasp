from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import base64
import os
from werkzeug.utils import secure_filename
import pandas as pd

app = Flask(__name__)

# Configuración de Telegram
TOKEN = "8107580499:AAG3FyXhtmXSPRb0To3hgZCa3WTTQm9Wfbo"
CHAT_ID = "-1002221266716"

# Ruta al modelo
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'modelo', 'impresion.pt')
model = None

# Cargar modelo
def load_model():
    global model
    try:
        print(f"Cargando modelo desde: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        print("Modelo cargado correctamente")
        return True
    except Exception as e:
        print(f"Error al cargar modelo: {e}")
        return False

def send_telegram_alert(image, detections):
    """Envía alerta a Telegram con imagen y detecciones (excepto 'imprimiendo')"""
    try:
        filtered_detections = detections[detections['name'].str.lower() != 'imprimiendo']
        if filtered_detections.empty:
            print("No se envía alerta: solo se detectó 'imprimiendo'")
            return {"status": "no_alert", "message": "Estado normal"}

        is_success, buffer = cv2.imencode(".jpg", image)
        if not is_success:
            return {"status": "error", "message": "Error al codificar imagen"}

        photo_bytes = BytesIO(buffer)
        photo_bytes.seek(0)
        files = {'photo': ('detection.jpg', photo_bytes)}

        message = "⚠ Detección de error en impresión 3D ⚠\n\n"
        for _, row in filtered_detections.iterrows():
            message += f"🔹 {row['name']}\n"
            message += f"Confianza: {row['confidence']:.2f}\n"
            message += f"Posición: x1={row['xmin']:.0f}, y1={row['ymin']:.0f}, x2={row['xmax']:.0f}, y2={row['ymax']:.0f}\n\n"

        url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
        data = {"chat_id": CHAT_ID, "caption": message, "parse_mode": "Markdown"}
        response = requests.post(url, data=data, files=files)

        if response.status_code == 200:
            print("Alerta enviada a Telegram exitosamente")
            return {"status": "success", "message": "Alerta enviada"}
        else:
            print(f"Error al enviar a Telegram: {response.text}")
            return {"status": "error", "message": f"Error Telegram: {response.text}"}
    except Exception as e:
        print(f"Error en send_telegram_alert: {e}")
        return {"status": "error", "message": str(e)}

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "running",
        "message": "Servidor de detección 3D activo",
        "model_loaded": model is not None
    })

@app.route('/detect', methods=['POST'])
def detect_errors():
    try:
        if model is None:
            return jsonify({"error": "Modelo no cargado"}), 500

        if 'image' not in request.files:
            return jsonify({"error": "No se envió imagen"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Nombre de archivo vacío"}), 400

        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "No se pudo decodificar la imagen"}), 400

        # Realizar detección
        results = model(frame)[0]
        boxes = results.boxes
        names = model.names

        detections_list = []
        error_count = 0
        data = []

        for box in boxes:
            cls_id = int(box.cls[0])
            name = names[cls_id]
            conf = float(box.conf[0])
            coords = box.xyxy[0].cpu().numpy()
            xmin, ymin, xmax, ymax = coords

            data.append({
                "name": name,
                "confidence": conf,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax
            })

            if name.lower() != 'imprimiendo':
                error_count += 1

            detections_list.append({
                "name": name,
                "confidence": conf,
                "coordinates": {
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax
                }
            })

        # Renderizar imagen con resultados
        result_img = results.plot()

        # Enviar alerta si hay errores
        df = pd.DataFrame(data)
        alert_result = {"status": "no_alert"}
        if error_count > 0:
            alert_result = send_telegram_alert(result_img, df)

        return jsonify({
            "status": "success",
            "detections": detections_list,
            "total_detections": len(detections_list),
            "error_count": error_count,
            "alert": alert_result
        })

    except Exception as e:
        print(f"Error en detect_errors: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({
        "server": "online",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH)
    })

if __name__ == '__main__':
    print("Iniciando servidor de detección 3D...")
    if load_model():
        print("✅ Modelo cargado correctamente")
    else:
        print("❌ Error al cargar modelo")

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
