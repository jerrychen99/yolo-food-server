from flask import Flask, request, jsonify
from ultralytics import YOLO
import tempfile
import os

app = Flask(__name__)
model = YOLO("best.pt")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        file.save(tmp.name)
        results = model(tmp.name)

    names = model.names
    detections = results[0].boxes

    food_info = {
        "apple": { "calories": 95, "protein": 1 },
        "banana": { "calories": 105, "protein": 1 },
        "sushi": { "calories": 300, "protein": 12 },
        "chicken breast": { "calories": 165, "protein": 31 },
        "tofu": { "calories": 76, "protein": 8 },
        "egg": { "calories": 78, "protein": 6 },
        "salmon": { "calories": 208, "protein": 20 },
        "steak": { "calories": 271, "protein": 25 },
        "cheeseburger": { "calories": 303, "protein": 17 },
        "pizza": { "calories": 285, "protein": 12 },
        "spaghetti": { "calories": 221, "protein": 8 },
        "fried rice": { "calories": 238, "protein": 6 },
        "pancake": { "calories": 86, "protein": 2 },
        "french fries": { "calories": 312, "protein": 4 },
        "broccoli": { "calories": 55, "protein": 4 },
        "carrot": { "calories": 41, "protein": 1 },
        "grapes": { "calories": 62, "protein": 1 },
        "orange": { "calories": 62, "protein": 1 },
        "milk": { "calories": 103, "protein": 8 },
        "yogurt": { "calories": 100, "protein": 10 }
    }

    if detections is None or len(detections) == 0:
        return jsonify({"food": "cannot detect", "calories": "?", "protein": "?"})

    for box in detections:
        cls_id = int(box.cls[0])
        food_name = names[cls_id].lower()

        if food_name in food_info:
            info = food_info[food_name]
            return jsonify({
                "food": food_name,
                "calories": info["calories"],
                "protein": info["protein"]
            })

    return jsonify({"food": "cannot detect", "calories": "?", "protein": "?"})
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port)
