from flask import Flask, request, jsonify
from ultralytics import YOLO
import tempfile

app = Flask(__name__)
model = YOLO("yolov8n.pt")  # 或你的模型名稱

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        file.save(tmp.name)
        results = model(tmp.name)

    names = model.names
    detections = results[0].boxes

    if detections is None or len(detections) == 0:
        return jsonify({"food": "無法辨識", "calories": "?", "protein": "?"})

    cls_id = int(detections.cls[0])
    food_name = names[cls_id]

    # 食物的熱量與蛋白質（範例）
    food_info = {
        "apple": {"calories": 95, "protein": 1},
        "banana": {"calories": 105, "protein": 2},
        "sushi": {"calories": 300, "protein": 12},
        # 你可以繼續加其他食物...
    }

    info = food_info.get(food_name.lower(), {"calories": "?", "protein": "?"})

    return jsonify({
        "food": food_name,
        "calories": info["calories"],
        "protein": info["protein"]
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050)
