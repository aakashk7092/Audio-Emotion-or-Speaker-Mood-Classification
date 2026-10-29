
from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from src.predict import load_prediction_resources, predict_emotion


st.set_page_config(
    page_title="Audio Emotion Classifier",
    page_icon="🎙️",
    layout="centered",
)

EMOTION_EMOJIS = {
    "neutral": "😐",
    "calm": "😌",
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "fearful": "😨",
    "disgust": "🤢",
    "surprised": "😲",
}


@st.cache_resource
def load_model_resources():
    return load_prediction_resources()


st.title("🎙️ Audio Emotion Classifier")
st.write(
    "Upload a speech audio recording to estimate its emotional tone."
)

st.info(
    "This is an experimental model. Predictions may be incorrect, "
    "especially for emotions with similar vocal characteristics."
)

try:
    model, mean, std = load_model_resources()
except (FileNotFoundError, ValueError, OSError) as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["wav"],
    help="Upload a WAV speech recording.",
)

if uploaded_file is not None:
    st.audio(uploaded_file.getvalue(), format="audio/wav")

    if st.button("Analyze Emotion", type="primary"):
        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = Path(temp_file.name)

            with st.spinner("Analyzing audio..."):
                result = predict_emotion(
                    temp_path,
                    model=model,
                    mean=mean,
                    std=std,
                )

            emotion = result["emotion"]
            confidence = result["confidence"]

            st.subheader("Prediction")
            st.markdown(
                f"## {EMOTION_EMOJIS.get(emotion, '🎭')} "
                f"{emotion.capitalize()}"
            )
            st.metric("Model confidence", f"{confidence:.2%}")

            st.subheader("Emotion probabilities")

            probabilities = result["probabilities"]
            probability_df = pd.DataFrame(
                {
                    "Emotion": list(probabilities.keys()),
                    "Probability": list(probabilities.values()),
                }
            ).sort_values("Probability", ascending=False)

            st.bar_chart(
                probability_df.set_index("Emotion"),
                y="Probability",
            )

            st.caption(
                "Probabilities are model estimates, not calibrated "
                "measures of how certain the prediction is."
            )

        except (ValueError, OSError, RuntimeError) as error:
            st.error(f"Could not analyze this audio: {error}")

        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
