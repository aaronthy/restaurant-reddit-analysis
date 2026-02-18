import joblib
import numpy as np
import streamlit as st

MODEL_PATH = "models/text_classifier.joblib"
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Review Labeler", page_icon="💬")
st.title("💬 Restaurant Review Labeler")
st.caption("Type a review like ChatGPT → get label + top guesses.")

if "history" not in st.session_state:
    st.session_state.history = []  # list of (user_text, pred, top3)

def top_k(model, text, k=3):
    scores = model.decision_function([text])[0]
    classes = model.classes_
    idx = np.argsort(scores)[::-1][:k]
    return [(classes[i], float(scores[i])) for i in idx]

# Chat input (press Enter to send)
user_text = st.chat_input("Paste a review and press Enter...")

if user_text:
    pred = model.predict([user_text])[0]
    top3 = top_k(model, user_text, k=3)
    st.session_state.history.append((user_text, pred, top3))

# Render chat history
for user_text, pred, top3 in st.session_state.history:
    with st.chat_message("user"):
        st.write(user_text)

    with st.chat_message("assistant"):
        st.write(f"**Predicted label:** {pred}")
        st.write("**Top-3 guesses:**")
        for lab, score in top3:
            st.write(f"- {lab} (score: {score:.3f})")
