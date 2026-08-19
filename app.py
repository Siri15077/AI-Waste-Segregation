from flask import Flask, render_template, request, send_from_directory
from predict import predict_waste
import os

app = Flask(__name__)
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    if "image" not in request.files:
        return render_template(
            "index.html",
            error="Please upload an image first."
        )

    image = request.files["image"]

    if image.filename == "":
        return render_template(
            "index.html",
            error="Please select an image."
        )

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        image.filename
    )

    image.save(image_path)

    try:

        waste_type, confidence = predict_waste(image_path)

        tips = {
            "plastic": "Place plastic items in the recyclable waste category.",
            "paper": "Keep paper clean and dry, then place it with recyclable paper waste.",
            "metal": "Place metal items in the recyclable metal waste category.",
            "glass": "Handle glass carefully and place it in the appropriate glass recycling category.",
            "cardboard": "Flatten cardboard and place it with recyclable paper/cardboard waste.",
            "trash": "This item could not be identified as one of the recyclable categories."
        }

        tip = tips.get(
            waste_type,
            "Dispose of this item according to your local waste guidelines."
        )

        return render_template(
            "index.html",
            result=waste_type,
            confidence=round(confidence, 2),
            tip=tip,
            image_file=image.filename
        )

    except Exception as e:

        return render_template(
            "index.html",
            error=f"AI Error: {str(e)}"
        )


if __name__ == "__main__":
app.run(host="0.0.0.0", port=5000)
