from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


app = Flask(__name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "bmp"}

# Load TFLite Model
interpreter = tf.lite.Interpreter(model_path="model/FreshGuard_AI.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Class names (same order as training)
class_names = [
    "freshapples",
    "freshbanana",
    "freshbittergroud",
    "freshcapsicum",
    "freshcucumber",
    "freshokra",
    "freshoranges",
    "freshpotato",
    "freshtomato",
    "rottenapples",
    "rottenbanana",
    "rottenbittergroud",
    "rottencapsicum",
    "rottencucumber",
    "rottenokra",
    "rottenoranges",
    "rottenpotato",
    "rottentomato"
]
def allowed_file(filename):

    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No image file provided", 400


    file = request.files["image"]

    filename_raw = file.filename or ""
    if filename_raw == "":
        return "No image selected", 400

    filename = secure_filename(filename_raw)


    # Save uploaded image
    upload_folder = "static/uploads"
    os.makedirs(upload_folder, exist_ok=True)

    upload_path = os.path.join(upload_folder, filename)

    file.save(upload_path)


    try:
        # Preprocessing
        img = load_img(
        upload_path,
        target_size=(224, 224)
    )

        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

    except Exception as e:
        return f"""
        <h2>❌ Unable to process image</h2>
        <p>{str(e)}</p>
        <br>
        <a href="/">⬅ Back</a>
        """


    # Prediction
    # Run prediction using TFLite
    interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]
    food_name = predicted_class.replace("fresh", "").replace("rotten", "")

    condition = "Fresh" if "fresh" in predicted_class else "Rotten"

    if condition == "Fresh":
        recommendation = "This item appears fresh and should be safe to use."
    else:
        recommendation = "This item appears rotten and should be discarded."

    confidence = float(prediction[0][predicted_index]) * 100
    
    # Determine confidence color based on confidence level
    if confidence >= 80:
        confidence_color = "green"
    elif confidence >= 50:
        confidence_color = "orange"
    else:
        confidence_color = "red"
    
    top3_indices = np.argsort(prediction[0])[-3:][::-1]

    top3_predictions = []
    for i in top3_indices:
        top3_predictions.append({
            "class": class_names[i],
            "confidence": round(float(prediction[0][i]) * 100, 2)
        })

    return render_template(
    "result.html",
    prediction=predicted_class,
    food_name=food_name,
    condition=condition,
    recommendation=recommendation,
    confidence=confidence,
    confidence_color=confidence_color,
    image_path=upload_path,
    top3_predictions=top3_predictions
)

    
    



if __name__ == "__main__":
    app.run(debug=True)