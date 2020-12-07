from flask import Flask, json, g, request, jsonify
import runner

app = Flask(__name__)

@app.route("/evaluate", methods=["POST"])
def evaluate():
    json_data = json.loads(request.data)
    input_text = json_data["query"]
    result = runner.evaluate(input_text)
    return jsonify(result=result)
