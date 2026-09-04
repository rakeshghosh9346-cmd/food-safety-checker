"""
Food Safety Checker - AI-powered tool to check if food items
(meat, fish, egg, vegetables, etc.) are safe to eat from a photo.

Uses Groq API with Qwen vision model (no dataset/training required).
Free API key, no card needed: https://console.groq.com
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

MODEL_NAME = "qwen/qwen3.6-27b"
MAX_IMAGE_SIZE = (512, 512)

PROMPT = """
Look at the food image (meat, fish, egg, vegetable, fruit, etc.) and judge if
it is safe to eat.

Reply in EXACTLY this short format, nothing else. No thinking, no analysis
steps, no extra explanation - just the final answer below:

Verdict: Safe / Not Safe / Uncertain
Why: one short sentence with the main visible reason
Risk: Low / Medium / High

Keep the whole reply under 5 lines total.
"""

# ----------------------------
# SESSION STATE
# ----------------------------
if "request_count" not in st.session_state:
    st.session_state.request_count = 0
if "cache" not in st.session_state:
    st.session_state.cache = {}

# ----------------------------
# API KEY (from Streamlit Secrets)
# ----------------------------
api_key = st.secrets.get("GROQ_API_KEY", None)

st.sidebar.title("Settings")

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
        else:
            try:
                resized_image = image.copy()
                resized_image.thumbnail(MAX_IMAGE_SIZE)

                buf = io.BytesIO()
                resized_image.save(buf, format="JPEG")
                img_bytes = buf.getvalue()
                b64_str = base64.b64encode(img_bytes).decode("utf-8")
                image_data_url = f"data:image/jpeg;base64,{b64_str}"

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
                            max_completion_tokens=300,
                        )

                        result_text = completion.choices[0].message.content
                        # Keep only the final answer - cut everything before "Verdict:"
                        # This removes any hidden reasoning/thinking text some models add
                        if "Verdict:" in result_text:
                            result_text = "Verdict:" + result_text.split("Verdict:", 1)[1]
                        result_text = result_text.replace("</think>", "").replace("<think>", "").strip()
                        st.session_state.cache[cache_key] = result_text
                        st.session_state.request_count += 1

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
