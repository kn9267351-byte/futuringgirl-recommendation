
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Cho phép website FuturingGirl gọi API
CORS(app, resources={
    r"/recommend": {
        "origins": ["https://futuringgirl.store"]
    }
})

# Đọc mô hình đã huấn luyện từ Kaggle
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "recommendation_model.joblib"

model = joblib.load(MODEL_PATH)

tfidf_matrix = model["tfidf_matrix"]
df = model["products"]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "FuturingGirl Recommendation API is running"
    })


@app.route("/recommend", methods=["GET"])
def recommend():
    product_id = request.args.get("product_id")
    top_n = request.args.get("top_n", default=5)

    try:
        product_id = int(product_id)
        top_n = int(top_n)

        if not 1 <= top_n <= 20:
            return jsonify({
                "error": "top_n phải từ 1 đến 20"
            }), 400

    except (TypeError, ValueError):
        return jsonify({
            "error": "product_id và top_n phải là số hợp lệ"
        }), 400

    # Tìm sản phẩm theo WooCommerce ID
    product_indices = df.index[df["id"] == product_id].tolist()

    if not product_indices:
        return jsonify({
            "error": "Không tìm thấy sản phẩm"
        }), 404

    product_idx = product_indices[0]

    # Tính độ tương đồng với các sản phẩm còn lại
    scores = cosine_similarity(
        tfidf_matrix[product_idx],
        tfidf_matrix
    ).flatten()

    # Sắp xếp theo độ tương đồng giảm dần
    similar_indices = np.argsort(scores)[::-1]
    similar_indices = [
        idx for idx in similar_indices
        if idx != product_idx
    ]

    top_indices = similar_indices[:top_n]

    recommendations = []

    for idx in top_indices:
        row = df.iloc[idx]

        recommendations.append({
            "id": int(row["id"]),
            "sku": str(row["sku"]),
            "name": str(row["name"]),
            "category": str(row["category"]),
            "price": str(row["price"]),
            "image_url": str(row["image_url"]),
            "product_url": str(row["product_url"]),
            "in_stock": bool(row["in_stock"]),
            "similarity_score": round(float(scores[idx]), 4)
        })

    return jsonify({
        "product_id": product_id,
        "total": len(recommendations),
        "recommendations": recommendations
    })


if __name__ == "__main__":
    app.run(debug=True)
