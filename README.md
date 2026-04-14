# Real-Time ASL Recognition Web Dashboard 🤟

A high-performance, real-time American Sign Language (ASL) translator. This system utilizes a two-stage Neural Network pipeline to track human hand gestures through a webcam, securely translate them locally into text, and stream the analytics to a beautiful, glassmorphism-styled Web UI.

## Core Features
*   **Real-Time Translation**: Achieves blisteringly fast inference (<50ms) using a lightweight PyTorch Multi-Layer Perceptron (MLP).
*   **Temporal Smoothing Buffer**: Completely eradicates visual flickering by enforcing a 5-frame identical prediction threshold before typing.
*   **Smart Auto-Spacing**: Automatically detects when hands leave the frame for >1 second to intelligently insert spaces.
*   **Customizable Confidence Threshold**: A dashboard slider allows you to aggressively filter out low-confidence AI predictions on the fly.
*   **Hardware Optimized Native Mount**: Safely unmount the camera sensor and pause the video stream natively to completely suspend all ML models, saving battery natively without closing the app.

## Project Architecture
1.  **Google MediaPipe Vision Tasks**: Extracts exactly 21 3D landmarks of hand coordinates natively from the raw video feed, stripping away all background noise.
2.  **Coordinate Normalizing Pipeline**: Shifts math tracking so it is completely relative to the wrist `(0, 0)`. The AI therefore works flawlessly regardless of distance from the camera or individual hand proportions.
3.  **Custom PyTorch Classifier**: A rapid 2-hidden-layer MLP network built from scratch and trained directly on localized hand geometry to output Softmax configurations corresponding to ASL Letters.
4.  **Flask Global State Streaming**: Locks internal application state globally inside a generator to prevent duplicate Ghost Threads. Multiple sockets/users can poll the AI dynamically without UI race conditions.

## How to Run Locally

1. **Clone the repository** and open the project directory.
2. **Create a python virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Launch the Flask Server**:
   ```bash
   python3 src/web.py
   ```
5. **Open Browser**: Navigate to `http://localhost:5001`.

## Retraining Instructions

If landmark rows (CSV data) is ready, then follow this:
1. Place your CSV data in `/data/`.
2. Normalize coordinates and format logic: `python3 src/data_prep.py --input_csv data/your_dataset.csv`
3. Hit re-compile on the deep learning network: `python3 src/train.py`
The system will instantly output an optimized `asl_mlp.pth`. Restart your Web Server to utilize your new intelligence!

For images files(png, jpg and jpeg) from Kaggle Dataset, create landmark rows(csv file) first.
1. Place your image files in `/data/images/`.
2. Create landmark csv file `python3 src/data_prep.py --image_dir data/images --extract_csv data/landmarks.csv`
3. Normalize coordinates and format logic: `python3 src/data_prep.py --input_csv data/landmarks.csv`
4. Hit re-compile on the deep learning network: `python3 src/train.py`
The system will instantly output an optimized `asl_mlp.pth`. Restart your Web Server to utilize your new intelligence!