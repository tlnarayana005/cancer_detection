import streamlit as st
from PIL import Image

from src.inference.predictor import CancerPredictor
from src.utils.config import load_config
from src.utils.logger import create_logger
from src.visualization.plots import plot_gradcam_image


def main():
    st.set_page_config(page_title="Cancer Detection AI", layout="centered")
    st.title("Cancer Detection AI")
    st.write(
        "Upload a medical image and get a cancer prediction with Grad-CAM explainability."
    )

    config = load_config("configs/config.yaml")
    logger = create_logger("app.log")
    predictor = CancerPredictor(config=config, logger=logger)

    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    use_checkpoint = st.text_input(
        "Optional checkpoint path", value="outputs/checkpoints/best_model.h5"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_column_width=True)

        if st.button("Predict"):
            with st.spinner("Running inference..."):
                label, confidence, gradcam_path = predictor.predict_with_explainability(
                    uploaded_file, checkpoint_path=use_checkpoint
                )
                st.success(f"Prediction: {label} ({confidence:.2%})")
                if gradcam_path:
                    st.image(gradcam_path, caption="Grad-CAM visualization", use_column_width=True)

    st.sidebar.header("About")
    st.sidebar.write(
        "This app uses transfer learning and Grad-CAM to explain predictions for cancer detection from medical images."
    )


if __name__ == "__main__":
    main()
