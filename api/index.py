from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow CORS so Wix can access it

# ✅ Home route (for testing)
@app.route('/')
def home():
    return jsonify({"message": "Model API working!"})

# ✅ Prediction route (main endpoint)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' in request"}), 400

        text = data["text"]

        # ⚙️ Example logic (you can replace this with your model inference)
        result = f"Prediction for '{text}'"

        # Send the result back
        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


#
