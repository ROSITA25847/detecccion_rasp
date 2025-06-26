from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import base64
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuración de Telegram
TOKEN = "8107580499:AAG3FyXhtmXSPRb0To3hgZCa3WTTQm9Wfbo"
CHAT_ID = "-1002221266716"

# Cargar modelo al iniciar la aplicación
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'modelo', 'impresion.pt')
model = None

def load_model():
    global model
    try:
        print(f"Cargando modelo desde: {MODEL_PATH}")
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)
        # Optimizar modelo
        model.conf = 0.25
        model.iou = 0.45
        model.agnostic = False
        model.multi_label = False
        model.max_det = 1000
        print("Modelo cargado y optimizado correctamente")
        return True
    except Exception as e:
        print(f"Error al cargar modelo: {e}")
        return False

def send_telegram_alert(image, detections):
    """Envía alerta a Telegram con imagen y detecciones (excepto 'imprimiendo')"""
    try:
        # Filtrar detecciones para excluir 'imprimiendo'
        filtered_detections = detections[detections['name'].str.lower() != 'imprimiendo']
        
        if filtered_detections.empty:
            print("No se envía alerta: solo se detectó 'imprimiendo' (estado normal)")
            return {"status": "no_alert", "message": "Estado normal"}
        
        # Convertir imagen a bytes
        is_success, buffer = cv2.imencode(".jpg", image)
        if not is_success:
            return {"status": "error", "message": "Error al codificar imagen"}

        # Preparar archivos para Telegram
        photo_bytes = BytesIO(buffer)
        photo_bytes.seek(0)
        files = {'photo': ('detection.jpg', photo_bytes)}

        # Crear mensaje
        message = "⚠ Detección de error en impresión 3D ⚠\n\n"
        for _, row in filtered_detections.iterrows():
            message += f"🔹 {row['name']}\n"
            message += f"Confianza: {row['confidence']:.2f}\n"
            message += f"Posición: x1={row['xmin']:.0f}, y1={row['ymin']:.0f}, x2={row['xmax']:.0f}, y2={row['ymax']:.0f}\n\n"

        # Enviar a Telegram
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
    """Endpoint de verificación de salud"""
    return jsonify({
        "status": "running",
        "message": "Servidor de detección 3D activo",
        "model_loaded": model is not None
    })

@app.route('/detect', methods=['POST'])
def detect_errors():
    """Endpoint principal para detección de errores"""
    try:
        if model is None:
            return jsonify({"error": "Modelo no cargado"}), 500
        
        # Verificar que se envió una imagen
        if 'image' not in request.files:
            return jsonify({"error": "No se envió imagen"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Nombre de archivo vacío"}), 400
        
        # Leer y procesar imagen
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "No se pudo decodificar la imagen"}), 400
        
        # Realizar detección
        detect = model(frame)
        info = detect.pandas().xyxy[0]
        
        # Generar imagen con detecciones
        result_img = np.squeeze(detect.render())
        
        # Preparar respuesta
        detections_list = []
        error_count = 0
        
        if len(info) > 0:
            for _, row in info.iterrows():
                detection = {
                    "name": row['name'],
                    "confidence": float(row['confidence']),
                    "coordinates": {
                        "xmin": float(row['xmin']),
                        "ymin": float(row['ymin']),
                        "xmax": float(row['xmax']),
                        "ymax": float(row['ymax'])
                    }
                }
                detections_list.append(detection)
                
                # Contar errores (excluyendo 'imprimiendo')
                if row['name'].lower() != 'imprimiendo':
                    error_count += 1
        
        # Enviar alerta si hay errores
        alert_result = {"status": "no_alert"}
        if error_count > 0:
            alert_result = send_telegram_alert(result_img, info)
        
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
    """Endpoint para verificar estado del servidor"""
    return jsonify({
        "server": "online",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH)
    })

if __name__ == '__main__':
    print("Iniciando servidor de detección 3D...")
    
    # Cargar modelo
    if load_model():
        print("✅ Modelo cargado correctamente")
    else:
        print("❌ Error al cargar modelo")
    
    # Obtener puerto de Render o usar 5000 por defecto
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)