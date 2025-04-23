from flask import Flask, render_template, request
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from src.translation.translator import translate_english_to_urdu
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    translation = ""
    if request.method == "POST":
        english_text = request.form["english_text"]
        if english_text.strip() != "":
            translation = translate_english_to_urdu(english_text)
    return render_template("index.html", translation=translation)

if __name__ == "__main__":
    app.run(debug=True)