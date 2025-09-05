import os
from flask import Flask, request, render_template, session, redirect, url_for, flash
import google.generativeai as genai

app = Flask(__name__)
app.secret_key = "super_secret_key"  # For session

UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Prompt styles
STYLE_PROMPTS = {
    "shashi": "Describe this image in the eloquent, polysyllabic, and grandiloquent style of Shashi Tharoor.Limit the description to 150 words.",
    "shakespeare": "Describe this image in the poetic and antiquated language of William Shakespeare.Limit the description to 150 words.",
    "casual": "Describe this image in a casual, relaxed, and friendly tone like you're talking to a friend.Limit the description to 150 words.",
    "genz": "Describe this image in Gen Z style using modern slang, emojis, and meme language. Be playful and expressive.Limit the description to 150 words."
}


@app.route("/", methods=["GET", "POST"])
def index():
    caption = None
    selected_style = session.get("selected_style", "casual")

    if request.method == "POST":
        # Ensure API key is set
        api_key = session.get("gemini_api_key")
        if not api_key:
            flash("Please set your Gemini API key in the sidebar.")
            return redirect(url_for("index"))

        file = request.files.get("image")
        selected_style = request.form.get("style", "casual")
        session["selected_style"] = selected_style

        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            session["image_path"] = filepath
            return redirect(url_for("caption"))

    return render_template("index.html", image_url=None, caption=None, selected_style=selected_style)


@app.route("/caption", methods=["GET", "POST"])
def caption():
    image_path = session.get("image_path")
    selected_style = session.get("selected_style", "casual")
    api_key = session.get("gemini_api_key")
    caption = None

    if not image_path or not api_key:
        flash("Please upload an image and set your Gemini API key.")
        return redirect(url_for("index"))

    if request.method == "POST":
        selected_style = request.form.get("style", "casual")
        session["selected_style"] = selected_style

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        with open(image_path, "rb") as img_file:
            image_bytes = img_file.read()

        prompt = STYLE_PROMPTS[selected_style]

        response = model.generate_content([
            prompt,
            {
                "mime_type": "image/jpeg",
                "data": image_bytes
            }
        ])

        caption = response.text.strip()

    except Exception as e:
        caption = f"❌ Error generating caption: {str(e)}"

    return render_template("index.html", image_url=image_path, caption=caption, selected_style=selected_style)


@app.route("/set-key", methods=["POST"])
def set_key():
    key = request.form.get("api_key")
    if key:
        session["gemini_api_key"] = key.strip()
        flash("✅ Gemini API key saved successfully.")
    else:
        flash("⚠️ Please enter a valid API key.")
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000,debug=True)