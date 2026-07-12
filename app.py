
import scrapping as scp
from deep_translator import GoogleTranslator
from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data["message"]
    language = data.get("language", "en")
    if language == "rw":
        english_message = GoogleTranslator(source="rw",target="en").translate(message)

        answer = scp.answer_finder(english_message)

        answer = GoogleTranslator(source="en",target="rw").translate(answer)
    else:
        answer = scp.answer_finder(message)
    
    answer = (
    answer
    .replace("â€“", "-")
    .replace("â€”", "-")
    .replace("â€", "")
    )
    return jsonify({
        "response": answer
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
