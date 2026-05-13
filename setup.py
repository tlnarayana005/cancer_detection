from setuptools import find_packages, setup

setup(
    name="cancer_detection_ai",
    version="0.1.0",
    author="AI Engineer",
    author_email="contact@example.com",
    description="Cancer detection project using deep learning, transfer learning, and explainability.",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=[
        "tensorflow>=2.14.0",
        "opencv-python>=4.8.0",
        "streamlit>=1.25.0",
        "scikit-learn>=1.4.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "pillow>=10.0.0",
        "pandas>=2.1.0",
        "loguru>=0.7.0",
    ],
    entry_points={
        "console_scripts": [
            "cancer-train=train:main",
            "cancer-predict=predict:main",
            "cancer-app=app:main",
        ],
    },
)
