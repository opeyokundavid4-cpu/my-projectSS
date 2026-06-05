# ASL Translation System MVP

This project is a Minimum Viable Product (MVP) for an American Sign Language (ASL) translation system using a Convolutional Neural Network (CNN).

## Features
- **Sign Recognition**: Recognizes basic ASL alphabets (A, B, C, D, E).
- **Real-time Detection**: Captures input from a webcam and provides instant translation.
- **User Interface**: A simple web-based interface for interaction and sentence building.
- **High Accuracy**: Trained on a diverse synthetic dataset with high precision.

## Project Structure
- `app/`: Contains the Flask backend.
- `data/`: Dataset and preprocessed files.
- `models/`: Trained CNN model.
- `scripts/`: Data generation, training, and evaluation scripts.
- `static/`: Frontend files (HTML/JS) and evaluation reports.

## How to Run
1. Install dependencies:
   ```bash
   pip install tensorflow opencv-python-headless flask flask-cors numpy scikit-learn matplotlib seaborn
   ```
2. Navigate to the `app` directory and start the server:
   ```bash
   cd asl_mvp/app
   python main.py
   ```
3. Open the application in your browser at `http://localhost:5000`.

## Evaluation
The model achieved 100% accuracy on the synthetic test set. Detailed metrics and confusion matrices can be found in the `static/` directory.
