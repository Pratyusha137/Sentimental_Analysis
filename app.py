import streamlit as st
import base64
import pickle
import os

# 🔹 Page config
st.set_page_config(page_title="Sentiment Analysis App", page_icon="🎬")

# 🔹 Left-align everything
st.markdown("""
<style>
/* Left align main content */
.block-container {
    padding-left: 3rem;
    padding-right: 3rem;
    text-align: left;
}
</style>
""", unsafe_allow_html=True)

# 🔹 Set background image and styles
def set_background(image_file):
    # Try relative path first
    if not os.path.exists(image_file):
        # Try current working directory
        image_file = os.path.join(os.getcwd(), image_file)
    
    if os.path.exists(image_file):
        with open(image_file, "rb") as image:
            encoded = base64.b64encode(image.read()).decode()
        css = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
        }}
        .stTextInput > div > div > input,
        .stTextArea textarea {{
            background-color: rgba(0, 0, 0, 0.6);
            color: yellow;
            font-size: 16px;
            border-radius: 8px;
        }}
        .stButton > button {{
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.5em 1em;
        }}
        .custom-output {{
            color: white;
            font-size: 20px;
            font-weight: bold;
        }}
        footer {{
            visibility: hidden;
        }}
        footer:after {{
            content: "🔸 Made with Streamlit";
            visibility: visible;
            display: block;
            text-align: center;
            color: yellow;
            padding: 10px;
            font-size: 14px;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ Background image not found: {image_file}")

# 🔹 Set background using absolute path
set_background(r"D:\Sentimental_analysis\Sentimental_Analysis\Movie_Background.jpg")

# 🔹 Title and subtitle
st.markdown("""
<h1 style='color: White;'>🎬 Movie Review Sentiment Analysis</h1>
<p style='color: Black; font-size: 18px;'>
    Enter a movie review below to find out whether it's <b>Positive</b> or <b>Negative</b>!
</p>
""", unsafe_allow_html=True)

# 🔹 Load model and vectorizer
with open('random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# 🔹 Input box
st.markdown("<h5 style='color: yellow;'>Write your review here:</h5>", unsafe_allow_html=True)
user_input = st.text_area("", height=200)

# 🔹 Predict button
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)[0]
        if prediction == 'positive':
            st.markdown('<div class="custom-output">🌟 Predicted Sentiment: Positive</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="custom-output">☹️ Predicted Sentiment: Negative</div>', unsafe_allow_html=True)
