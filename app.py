from flask import Flask, request, render_template_string
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import pandas as pd
import base64
from io import BytesIO

app = Flask(__name__)

# =====================================================
# LOAD DATASETS
# =====================================================

fashion_df = pd.read_csv("outfits.csv")

shopping_df = pd.read_csv("shopping.csv")

# =====================================================
# HTML TEMPLATE
# =====================================================

HTML = """

<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>AuraFit AI</title>

<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<style>

*{
    margin:0;
    padding:0;
    box-sizing:border-box;
    font-family:'Poppins',sans-serif;
}

body{

    background:
    linear-gradient(
        135deg,
        #0f172a,
        #111827,
        #1e293b
    );

    color:white;

    min-height:100vh;

    overflow-x:hidden;
}

/* ===========================================
BACKGROUND GLOW
=========================================== */

body::before{

    content:"";

    position:fixed;

    width:500px;
    height:500px;

    background:
    radial-gradient(
        rgba(166,193,238,0.12),
        transparent
    );

    top:-200px;
    left:-200px;

    z-index:-1;
}

/* ===========================================
NAVBAR
=========================================== */

.navbar{

    width:100%;

    padding:25px 8%;

    display:flex;

    justify-content:space-between;

    align-items:center;
}

.logo{

    font-size:32px;

    font-weight:700;

    background:
    linear-gradient(
        to right,
        #f9c5d1,
        #fbc2eb
    );

    -webkit-background-clip:text;

    -webkit-text-fill-color:transparent;
}

/* ===========================================
HERO
=========================================== */

.hero{

    width:90%;

    margin:auto;

    margin-top:40px;

    display:flex;

    gap:50px;

    flex-wrap:wrap;
}

.hero-left{

    flex:1;

    min-width:300px;
}

.hero-left h1{

    font-size:70px;

    line-height:1.1;

    margin-bottom:20px;
}

.gradient-text{

    background:
    linear-gradient(
        to right,
        #fbc2eb,
        #a6c1ee
    );

    -webkit-background-clip:text;

    -webkit-text-fill-color:transparent;
}

.hero-left p{

    color:#cbd5e1;

    font-size:18px;

    line-height:1.8;

    margin-bottom:35px;
}

/* ===========================================
RIGHT SIDE MODEL CARD
=========================================== */

.hero-right{

    flex:1;

    min-width:300px;

    display:flex;

    justify-content:center;

    align-items:center;
}

.fashion-card{

    width:400px;

    height:520px;

    border-radius:35px;

    overflow:hidden;

    background:
    rgba(255,255,255,0.08);

    border:
    1px solid rgba(255,255,255,0.1);

    backdrop-filter:blur(15px);

    box-shadow:
    0px 0px 50px rgba(255,255,255,0.05);
}

.fashion-card img{

    width:100%;

    height:100%;

    object-fit:cover;
}

/* ===========================================
UPLOAD CARD
=========================================== */

.upload-card{

    position:relative;

    background:
    rgba(255,255,255,0.08);

    border:
    1px solid rgba(255,255,255,0.15);

    backdrop-filter:blur(18px);

    border-radius:35px;

    padding:45px;

    overflow:hidden;
}

.upload-area{

    border:
    2px dashed rgba(255,255,255,0.2);

    border-radius:25px;

    padding:55px 20px;

    text-align:center;

    background:
    rgba(255,255,255,0.03);
}

.upload-icon{

    font-size:70px;

    margin-bottom:20px;
}

.upload-title{

    font-size:26px;

    font-weight:600;

    margin-bottom:12px;
}

.upload-subtitle{

    color:#cbd5e1;

    font-size:15px;

    margin-bottom:25px;

    line-height:1.8;
}

/* ===========================================
FILE INPUT
=========================================== */

input[type=file]{

    width:100%;

    padding:18px;

    border-radius:18px;

    background:#111827;

    color:white;

    border:none;
}

input[type=file]::file-selector-button{

    background:
    linear-gradient(
        to right,
        #fbc2eb,
        #a6c1ee
    );

    border:none;

    padding:12px 18px;

    border-radius:12px;

    color:black;

    font-weight:600;

    margin-right:15px;
}

/* ===========================================
BUTTON
=========================================== */

.analyze-btn{

    width:100%;

    margin-top:25px;

    padding:18px;

    border:none;

    border-radius:18px;

    background:
    linear-gradient(
        to right,
        #fbc2eb,
        #a6c1ee
    );

    color:black;

    font-size:18px;

    font-weight:700;

    cursor:pointer;
}

/* ===========================================
RESULT SECTION
=========================================== */

.result-section{

    width:90%;

    margin:auto;

    margin-top:60px;

    margin-bottom:60px;

    display:grid;

    grid-template-columns:
    repeat(auto-fit,minmax(320px,1fr));

    gap:40px;
}

.result-card{

    background:
    rgba(255,255,255,0.08);

    border:
    1px solid rgba(255,255,255,0.1);

    backdrop-filter:blur(15px);

    padding:35px;

    border-radius:30px;
}

.result-card h2{

    margin-bottom:20px;
}

.outfit-item{

    margin-top:18px;

    padding:18px;

    background:
    rgba(255,255,255,0.05);

    border-radius:18px;
}

.buy-btn{

    display:inline-block;

    margin-top:12px;

    padding:10px 16px;

    background:
    linear-gradient(
        to right,
        #fbc2eb,
        #a6c1ee
    );

    color:black;

    text-decoration:none;

    border-radius:12px;

    font-weight:600;
}

/* ===========================================
UPLOADED IMAGE
=========================================== */

.user-image{

    width:100%;

    border-radius:25px;

    margin-top:20px;
}

/* ===========================================
FOOTER
=========================================== */

.footer{

    text-align:center;

    padding:30px;

    color:#94a3b8;
}

</style>

</head>

<body>

<!-- ===================================== -->
<!-- NAVBAR -->
<!-- ===================================== -->

<div class="navbar">

<div class="logo">
AuraFit AI ✨
</div>

</div>

<!-- ===================================== -->
<!-- HERO -->
<!-- ===================================== -->

<div class="hero">

<div class="hero-left">

<h1>

Your Personal
<span class="gradient-text">
AI Fashion Stylist
</span>

</h1>

<p>

Upload your selfie and let AI analyze your style,
recommend premium outfits,
and find the cheapest places to buy them.

</p>

<!-- ===================================== -->
<!-- UPLOAD CARD -->
<!-- ===================================== -->

<div class="upload-card">

<form method="POST" enctype="multipart/form-data">

<div class="upload-area">

<div class="upload-icon">
✨
</div>

<div class="upload-title">

Upload Your Selfie

</div>

<div class="upload-subtitle">

AI-powered fashion recommendations,
shopping assistant,
and premium styling insights.

</div>

<input type="file" name="image" required>

<button type="submit" class="analyze-btn">

Analyze My Style →

</button>

</div>

</form>

</div>

</div>

<!-- ===================================== -->
<!-- RIGHT SIDE MODEL IMAGE -->
<!-- ===================================== -->

<div class="hero-right">

<div class="fashion-card">

<img src="https://images.unsplash.com/photo-1529139574466-a303027c1d8b?q=80&w=1000&auto=format&fit=crop">

</div>

</div>

</div>

<!-- ===================================== -->
<!-- RESULTS -->
<!-- ===================================== -->

{% if image %}

<div class="result-section">

<!-- ANALYSIS -->

<div class="result-card">

<h2>🧠 AI Style Analysis</h2>

<img
src="data:image/png;base64,{{ image }}"
class="user-image">

<p><b>Skin Tone:</b> {{ tone }}</p>

<p><b>Undertone:</b> {{ undertone }}</p>

<p><b>Fashion Style:</b> {{ style }}</p>

<p><b>Match Score:</b> {{ score }}%</p>

</div>

<!-- OUTFIT -->

<div class="result-card">

<h2>✨ Recommended Outfit</h2>

<div class="outfit-item">

<b>Top</b>

<p>{{ top }}</p>

</div>

<div class="outfit-item">

<b>Bottom</b>

<p>{{ bottom }}</p>

</div>

<div class="outfit-item">

<b>Shoes</b>

<p>{{ shoes }}</p>

</div>

<div class="outfit-item">

<b>Accessories</b>

<p>{{ accessories }}</p>

</div>

</div>

<!-- SHOPPING -->

<div class="result-card">

<h2>🛍 Cheapest Shopping Options</h2>

<div class="outfit-item">

<b>{{ top }}</b>

<p>
Best Price:
₹{{ top_price }}
</p>

<p>
Store:
{{ top_store }}
</p>

<a
href="{{ top_link }}"
target="_blank"
class="buy-btn">

Buy Now →

</a>

</div>

<div class="outfit-item">

<b>{{ bottom }}</b>

<p>
Best Price:
₹{{ bottom_price }}
</p>

<p>
Store:
{{ bottom_store }}
</p>

<a
href="{{ bottom_link }}"
target="_blank"
class="buy-btn">

Buy Now →

</a>

</div>

<div class="outfit-item">

<b>{{ shoes }}</b>

<p>
Best Price:
₹{{ shoe_price }}
</p>

<p>
Store:
{{ shoe_store }}
</p>

<a
href="{{ shoe_link }}"
target="_blank"
class="buy-btn">

Buy Now →

</a>

</div>

</div>

</div>

{% endif %}

<!-- ===================================== -->
<!-- FOOTER -->
<!-- ===================================== -->

<div class="footer">

AuraFit AI • AI Fashion Recommendation & Shopping Assistant

</div>

</body>

</html>

"""

# =====================================================
# SKIN TONE DETECTION
# =====================================================

def detect_skin_tone(image):

    img = np.array(image)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img = cv2.resize(img, (250,250))

    pixels = img.reshape((-1,3))

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_

    dominant = colors[0]

    r,g,b = dominant

    brightness = (r+g+b)/3

    if brightness > 180:

        tone = "Fair"
        undertone = "Cool"

    elif brightness > 130:

        tone = "Medium"
        undertone = "Neutral"

    else:

        tone = "Dusky"
        undertone = "Warm"

    return tone, undertone

# =====================================================
# RECOMMENDATION ENGINE
# =====================================================

def recommend_outfit(undertone, tone):

    if undertone == "Warm":

        filtered = fashion_df[
            fashion_df["SkinTone"] == "Warm"
        ]

    elif undertone == "Neutral":

        filtered = fashion_df[
            fashion_df["SkinTone"] == "Neutral"
        ]

    else:

        filtered = fashion_df[
            fashion_df["SkinTone"] == "Cool"
        ]

    filtered = filtered.copy()

    filtered["MatchScore"] = 95

    recommendation = filtered.sample(1).iloc[0]

    return recommendation

# =====================================================
# FIND CHEAPEST SHOP
# =====================================================

def find_cheapest(item_name):

    # ==========================================
    # EXACT MATCH
    # ==========================================

    exact_match = shopping_df[

        shopping_df["Item"].str.lower()

        == item_name.lower()
    ]

    if len(exact_match) > 0:

        cheapest = exact_match.sort_values(
            by="Price"
        ).iloc[0]

        return cheapest

    # ==========================================
    # SIMILAR MATCH
    # ==========================================

    keywords = item_name.lower().split()

    similar_products = pd.DataFrame()

    for word in keywords:

        matches = shopping_df[
            shopping_df["Item"]
            .str.lower()
            .str.contains(word)
        ]

        similar_products = pd.concat(
            [similar_products, matches]
        )

    similar_products = similar_products.drop_duplicates()

    if len(similar_products) > 0:

        cheapest = similar_products.sort_values(
            by="Price"
        ).iloc[0]

        return cheapest

    return None

# =====================================================
# HOME ROUTE
# =====================================================

@app.route("/", methods=["GET","POST"])

def home():

    if request.method == "POST":

        file = request.files["image"]

        image = Image.open(file)

        tone, undertone = detect_skin_tone(image)

        outfit = recommend_outfit(
            undertone,
            tone
        )

        # ==========================================
        # SHOPPING OPTIONS
        # ==========================================

        top_shop = find_cheapest(
            outfit["Top"]
        )

        bottom_shop = find_cheapest(
            outfit["Bottom"]
        )

        shoe_shop = find_cheapest(
            outfit["Shoes"]
        )

        # ==========================================
        # USER IMAGE
        # ==========================================

        buffered = BytesIO()

        image.save(buffered, format="PNG")

        img_str = base64.b64encode(
            buffered.getvalue()
        ).decode()

        return render_template_string(

            HTML,

            image=img_str,

            tone=tone,

            undertone=undertone,

            style=outfit["Style"],

            top=outfit["Top"],

            bottom=outfit["Bottom"],

            shoes=outfit["Shoes"],

            accessories=outfit["Accessories"],

            score=outfit["MatchScore"],

            top_store=top_shop["Store"] if top_shop is not None else "Not Available",
            top_price=top_shop["Price"] if top_shop is not None else "--",
            top_link=top_shop["Link"] if top_shop is not None else "#",

            bottom_store=bottom_shop["Store"] if bottom_shop is not None else "Not Available",
            bottom_price=bottom_shop["Price"] if bottom_shop is not None else "--",
            bottom_link=bottom_shop["Link"] if bottom_shop is not None else "#",

            shoe_store=shoe_shop["Store"] if shoe_shop is not None else "Not Available",
            shoe_price=shoe_shop["Price"] if shoe_shop is not None else "--",
            shoe_link=shoe_shop["Link"] if shoe_shop is not None else "#"

        )

    return render_template_string(HTML)

# =====================================================
# RUN APP
# =====================================================

if __name__ == "__main__":

    app.run(debug=True)