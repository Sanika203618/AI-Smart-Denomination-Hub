# app.py
from flask import Flask, render_template, request, jsonify
import random

app = Flask(__name__)

# --------------------------
# Denomination Model
# --------------------------
def calculate_denominations(amount, denominations=[500, 100, 20, 10, 5, 1]):
    distribution = {}
    remaining = amount
    for denom in denominations:
        count = remaining // denom
        if count > 0:
            distribution[denom] = int(count)
            remaining -= denom * count
    total = sum(k * v for k, v in distribution.items())
    return {"total": total, "distribution": distribution}

# --------------------------
# Budget Model
# --------------------------
def prepare_budget(expenses_input):
    """
    expenses_input = dict {category: amount}
    """
    total = sum(expenses_input.values())
    return {"total": total, "expenses": expenses_input}

# --------------------------
# Chatbot Model (Mock AI)
# --------------------------
def chatbot_response(message):
    # Simple fixed responses for demonstration
    message = message.lower().strip()
    if "budget" in message:
        return "You can enter your budget and categories in the Smart Budgeting page."
    elif "denomination" in message or "note" in message:
        return "Please use the Denomination Optimizer page to calculate optimal notes."
    elif "hello" in message or "hi" in message:
        return "Hello! How can I assist you with budgeting or denominations today?"
    else:
        # Random helpful tips for other messages
        tips = [
            "Did you know? Setting a budget can help you save more efficiently.",
            "Always try to withdraw exact denominations to avoid small change.",
            "Categorizing expenses makes budgeting easier!"
        ]
        return random.choice(tips)

# --------------------------
# Routes
# --------------------------

# Home Page
@app.route("/")
def index():
    return render_template("index.html")

# Denomination Page
@app.route("/denomination", methods=["GET", "POST"])
def denomination_page():
    result = None
    if request.method == "POST":
        try:
            amount = int(request.form.get("amount", 0))
            use_case = request.form.get("use_case", "General")
            result = calculate_denominations(amount)
            result["use_case"] = use_case
        except Exception as e:
            result = {"total": 0, "distribution": {}, "use_case": "Error"}
    return render_template("denomination.html", result=result)

# Budget Page
@app.route("/budget", methods=["GET", "POST"])
def budget_page():
    result = None
    if request.method == "POST":
        try:
            budget_input = float(request.form.get("budget_input", 0))
            categories = request.form.getlist("category")
            amounts = request.form.getlist("amount")
            expenses_input = {}
            for cat, amt in zip(categories, amounts):
                if cat.strip() != "" and amt.strip() != "":
                    expenses_input[cat.strip()] = float(amt)
            result = prepare_budget(expenses_input)
            result["budget"] = budget_input
        except Exception as e:
            result = {"total": 0, "expenses": {}}
    return render_template("budget.html", result=result)

# Chatbot Page
@app.route("/chatbot", methods=["GET"])
def chatbot_page():
    return render_template("chatbot.html")

# Chatbot Response
@app.route("/get_response", methods=["POST"])
def get_response():
    try:
        user_message = request.json.get("message", "")
        bot_reply = chatbot_response(user_message)
        return jsonify({"response": bot_reply})
    except:
        return jsonify({"response": "Sorry, something went wrong!"})

# --------------------------
# Run App
# --------------------------
if __name__ == "__main__":
    app.run(debug=True)
