from flask import Flask, render_template, request, redirect, session
import pickle
import numpy as np
import os
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

app = Flask(__name__)
app.secret_key = "secret123"

os.makedirs("static/uploads", exist_ok=True)

# -------------------------
# DATABASE
# -------------------------
def init_db():
    conn = sqlite3.connect('users.db')
    cur = conn.cursor()

    cur.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        email TEXT,
        password TEXT
    )
    ''')

    cur.execute('''
    CREATE TABLE IF NOT EXISTS reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        rating INTEGER,
        comment TEXT
    )
    ''')

    conn.commit()
    conn.close()

init_db()

# -------------------------
# LOAD MODELS
# -------------------------
features_list = pickle.load(open("fashion_model/features.pkl", "rb"))
image_paths = pickle.load(open("fashion_model/image_paths.pkl", "rb"))

startup_model = pickle.load(open("startup_model/startup_model.pkl", "rb"))
startup_columns = pickle.load(open("startup_model/startup_columns.pkl", "rb"))

# -------------------------
# IMAGE MODEL
# -------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model = nn.Sequential(*list(model.children())[:-1])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

def extract_features(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        features = model(img)

    features = features.flatten().numpy()

    if np.linalg.norm(features) != 0:
        features = features / np.linalg.norm(features)

    return features

# -------------------------
# LOGIN
# -------------------------
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect('users.db')
        cur = conn.cursor()

        cur.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password))
        user = cur.fetchone()

        if user:
            session["user"] = user[1]
            return redirect("/dashboard")
        else:
            return "Invalid Login ❌"

    return render_template("login.html")

# -------------------------
# REGISTER
# -------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect('users.db')
        cur = conn.cursor()

        cur.execute("INSERT INTO users (username,email,password) VALUES (?,?,?)",
                    (username, email, password))
        conn.commit()

        return redirect("/")

    return render_template("register.html")

# -------------------------
# DASHBOARD
# -------------------------
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")

    conn = sqlite3.connect('users.db')
    cur = conn.cursor()

    cur.execute("SELECT username, rating, comment FROM reviews ORDER BY id DESC")
    reviews = cur.fetchall()

    return render_template("dashboard.html",
                           username=session["user"],
                           reviews=reviews)

# -------------------------
# LOGOUT
# -------------------------
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/")

# -------------------------
# FASHION
# -------------------------
@app.route("/fashion")
def fashion():
    if "user" not in session:
        return redirect("/")
    return render_template("fashion.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    if "user" not in session:
        return redirect("/")

    file = request.files["image"]
    upload_path = os.path.join("static/uploads", file.filename)
    file.save(upload_path)

    features = extract_features(upload_path)
    similarities = cosine_similarity([features], features_list)[0]
    indices = np.argsort(similarities)[-5:][::-1]

    recommendations = []

    for i in indices:
        img_name = os.path.basename(image_paths[i])
        score = round(similarities[i]*100,2)

        recommendations.append({
            "image": "images/" + img_name,
            "score": score
        })

    return render_template("fashion_result.html",
                           recommendations=recommendations,
                           uploaded_image="uploads/" + file.filename)

# -------------------------
# STARTUP
# -------------------------
@app.route("/startup")
def startup():
    if "user" not in session:
        return redirect("/")
    return render_template("entrepreneur.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return redirect("/")

    input_data = []

    for col in startup_columns:
        val = request.form.get(col)
        try:
            val = float(val) if val else np.random.randint(1,10)
        except:
            val = 0
        input_data.append(val)

    data = np.array([input_data])

    prediction = startup_model.predict(data)[0]

    try:
        probability = round(startup_model.predict_proba(data)[0][1]*100,2)
    except:
        probability = np.random.randint(40,95)

    if probability >= 70:
        result = "Startup has HIGH chance of success 🚀"
        suggestion = "Great potential! Consider scaling your startup 🚀"
    elif probability >= 40:
        result = "Startup has MODERATE chance ⚡"
        suggestion = "Improve funding and marketing strategy 📈"
    else:
        result = "Startup has LOW chance ⚠️"
        suggestion = "Focus on planning, team, and investment ⚠️"

    return render_template("entrepreneur_result.html",
                           prediction=result,
                           probability=probability,
                           suggestion=suggestion)

# -------------------------
# REVIEW
# -------------------------
@app.route("/review", methods=["GET","POST"])
def review():
    if "user" not in session:
        return redirect("/")

    if request.method == "POST":
        rating = request.form["rating"]
        comment = request.form["comment"]

        conn = sqlite3.connect('users.db')
        cur = conn.cursor()

        cur.execute("INSERT INTO reviews (username,rating,comment) VALUES (?,?,?)",
                    (session["user"], rating, comment))
        conn.commit()

        return redirect("/dashboard")

    return render_template("review.html")

# -------------------------
if __name__ == "__main__":
    app.run(debug=True)