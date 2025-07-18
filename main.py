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

    # 假設我們只取第一個結果
    names = model.names
    detections = results[0].boxes

    if detections is None or len(detections) == 0:
        return jsonify({"food": "無法辨識", "calories": "?"})

    cls_id = int(detections.cls[0])
    food_name = names[cls_id]

    # 可以根據 food_name 給個預估熱量
    food_calories = {
        "apple": 95,
        "banana": 105,
        "sushi": 300,
        # ...
    }
    calories = food_calories.get(food_name.lower(), "?")

    return jsonify({
        "food": food_name,
        "calories": calories
    })


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050)
