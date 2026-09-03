"""
Food Safety Checker - AI-powered tool to check if food items
(meat, fish, egg, vegetables, etc.) are safe to eat from a photo.

Uses Groq API with Qwen vision model (no dataset/training required).
Free API key, no card needed: https://console.groq.com

Setup:
1. pip install streamlit groq pillow
2. Get a free API key from https://console.groq.com/keys
3. Run: streamlit run app.py
"""

import streamlit as st
from groq import Groq
from PIL import Image
import io
import base64

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Food Safety Checker", page_icon="🍗", layout="centered")

MODEL_NAME = "qwen/qwen3.6-27b"       # vision-capable model on Groq
MAX_IMAGE_SIZE = (512, 512)           # resize to save tokens
DAILY_LIMIT_PER_SESSION = 10          # simple per-session rate limit

PROMPT = """
You are a food safety inspection assistant. Look carefully at the image of
the food item provided (it could be meat, fish, egg, vegetable, fruit, or
another food item).

Answer STRICTLY in this format:

Verdict: <Safe / Not Safe / Uncertain>
Reason: <what visual signs led to this verdict - color, texture, spots, mold, discoloration, etc.>
Risk Level: <Low / Medium / High>
Note: <one line disclaimer that this is a screening tool, not a substitute for professional inspection>

Be concise. If the image is unclear or not a food item, say so in the Verdict.
"""

# ----------------------------
# SESSION STATE (basic rate limiting + cache)
# ----------------------------
if "request_count" not in st.session_state:
    st.session_state.request_count = 0
if "cache" not in st.session_state:
    st.session_state.cache = {}

# ----------------------------
# API KEY (loaded from Streamlit Secrets - user never sees or enters it)
# ----------------------------
api_key = st.secrets.get("GROQ_API_KEY", None)

st.sidebar.title("Settings")
st.sidebar.markdown(f"Requests used this session: {st.session_state.request_count}/{DAILY_LIMIT_PER_SESSION}")

# ----------------------------
# MAIN UI
# ----------------------------
st.title("🍗 Food Safety Checker")
st.write("Upload a photo of a food item (meat, fish, egg, vegetable, etc.) to check if it looks safe to eat.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Check Safety"):
        if not api_key:
            st.error("The app is not configured properly. Please contact the app owner.")
        elif st.session_state.request_count >= DAILY_LIMIT_PER_SESSION:
            st.warning("Daily limit reached for this session. Please try again later.")
        else:
            try:
                # ---- Resize image to save tokens ----
                resized_image = image.copy()
                resized_image.thumbnail(MAX_IMAGE_SIZE)

                # ---- Convert to base64 data URL (Groq needs this format) ----
                buf = io.BytesIO()
                resized_image.save(buf, format="JPEG")
                img_bytes = buf.getvalue()
                b64_str = base64.b64encode(img_bytes).decode("utf-8")
                image_data_url = f"data:image/jpeg;base64,{b64_str}"

                # ---- Simple cache key (based on image bytes hash) ----
                cache_key = hash(img_bytes)

                if cache_key in st.session_state.cache:
                    result_text = st.session_state.cache[cache_key]
                    st.info("Showing cached result for this exact image.")
                else:
                    with st.spinner("Analyzing image..."):
                        client = Groq(api_key=api_key)

                        completion = client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": PROMPT},
                                        {
                                            "type": "image_url",
                                            "image_url": {"url": image_data_url},
                                        },
                                    ],
                                }
                            ],
                            temperature=0.3,
                            max_completion_tokens=1024,
                        )

                        result_text = completion.choices[0].message.content
                        st.session_state.cache[cache_key] = result_text
                        st.session_state.request_count += 1

                # ---- Display result ----
                st.subheader("Result")
                st.markdown(result_text)

            except Exception as e:
                st.error("Something went wrong while analyzing the image. Please try again.")
                st.caption(f"Technical details: {e}")

st.markdown("---")
st.caption(
    "⚠️ This tool uses a general-purpose AI vision model and is intended as a "
    "screening aid only. It is not a substitute for professional food safety "
    "inspection or veterinary advice."
)
