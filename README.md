# AI-Based Sign Language Recognition System

An AI-powered application that recognizes hand signs and converts them into readable text, helping improve communication between people with hearing or speech impairments and others.

## Features

- Real-time sign recognition using a webcam
- Hand gesture recognition using a machine learning model
- Sign-to-text conversion
- Sentence formation from recognized signs
- Interactive web interface using Streamlit
- Model training using the Sign Language MNIST dataset

## Technologies Used

- Python
- Streamlit
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- MediaPipe
- Pillow

## Project Structure

```text
sign_language_project/
│
├── app.py
├── data_collection.py
├── train_model.py
├── test_model.py
├── webcam_test.py
├── model.pkl
├── sign_mnist_train.csv
├── sign_mnist_test.csv
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/megamithra/sign_language_project.git
```

2. Navigate to the project directory:

```bash
cd sign_language_project
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## Model Training

To train the machine learning model:

```bash
python train_model.py
```

The trained model is saved as `model.pkl`.

## Future Improvements

- Support for more sign language gestures
- Improved recognition accuracy
- Real-time sentence prediction
- Support for multiple sign languages
- Mobile application integration

## Author

**Megamithra**

GitHub: https://github.com/megamithra

---

*Developed as an AI and Machine Learning project.*
