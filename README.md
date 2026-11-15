# Digit Recognition Studio — Pencil & Indigo Edition

A Streamlit learning application for handwritten digit recognition with:
- A grey paper/pencil interface, dark-indigo typography and muted-gold borders.
- Animated indigo/dark-grey globe, splash transition and playful rocket launch.
- One to ten independent drawing slots, each with its own intended-digit selector.
- Image refinement that crops the ink, preserves the full drawing and centres it by intensity-weighted centre of mass.
- A CNN trained on MNIST, plus an automated validation agent.
- A clearer directional neural-network animation showing how the signal moves from pixels to class scores.
- A local sign-up/sign-in screen with salted PBKDF2 password hashes (prototype authentication only).
- An animated front-facing globe with a vertically spinning longitude grid, light and glow.

## 1. Environment

Recommended: Python 3.11–3.13 on Windows 11. TensorFlow is left unpinned above 2.20 so pip can select a build compatible with your Python version.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If TensorFlow installation fails, check that Python is 64-bit and use Python 3.11.

## 2. Train the CNN (one time)

```powershell
python scripts/train_model.py
```

The script downloads MNIST through Keras, trains a compact convolutional neural network, evaluates it on the held-out MNIST test set, and saves:

`models/mnist_digit_cnn.keras`

Training requires an internet connection on the first run. Model weights are intentionally not fabricated or represented as pretrained.

## 3. Run the app

```powershell
streamlit run app.py
```

## 4. How the automated agent decides

A sample is **ACCEPTED** only when:
- the canvas contains sufficient ink,
- no boundary clipping warning is present,
- it is not an exact duplicate in the current submission,
- CNN prediction matches the selected intended digit, and
- top-class confidence is at least 70%.

Otherwise the sample is **REJECTED** with findings explaining why. This is a conservative demonstration policy, not a guarantee that every accepted label is objectively correct. The confidence threshold should be calibrated on a separate validation set before real dataset collection.

## 5. Project structure

```text
digit_recognition_studio_indigo_pencil/
├── app.py
├── requirements.txt
├── README.md
├── modules/
│   ├── __init__.py
│   ├── refinery.py
│   ├── classifier.py
│   └── agent.py
├── scripts/
│   └── train_model.py
├── models/
│   └── (trained model is created here)
└── data/
    ├── accepted/
    ├── rejected/
    └── review_logs/
```

## 6. Learning note

The network used here is:

`28×28×1 → Conv(32) → MaxPool → Conv(64) → MaxPool → Flatten → Dense(128) → Dropout → Dense(10, Softmax)`

The animated neural-network graphic is conceptual. The actual numerical inference is performed by the saved Keras model.


## Local authentication note

The sign-up/sign-in screen is a local learning-project prototype. It stores a salted PBKDF2 password hash in `data/users.sqlite`, never a plain-text password. It does not provide email verification, password reset, multi-device identity, rate limiting, or production-grade account security. Do not use a primary/reused password. For a public deployment, replace this with a managed identity provider.

The landing page leads to a digit-shower/rocket transition, then the local sign-in/sign-up screen. Successful entry increments the canvas version so the drawing boards start clean.

## 7. Deploy on Streamlit Community Cloud

1. Train the model locally with `python scripts/train_model.py`.
2. Confirm that `models/mnist_digit_cnn.keras` exists. The model file must be committed to GitHub because the cloud container cannot access your computer's local files.
3. Create a GitHub repository and push the **contents** of this project folder (so `app.py` and `requirements.txt` are at the repository root).
4. Open <https://share.streamlit.io>, sign in with GitHub, choose **Create app**, then select your repository, branch `main`, and entrypoint `app.py`.
5. In **Advanced settings**, choose Python 3.13 to match the local environment and the TensorFlow version available in the requirements file.
6. Deploy and inspect the build logs if dependency installation fails.

The `.streamlit/config.toml` file carries the pencil/indigo theme into Community Cloud. Community Cloud installs dependencies from the unchanged `requirements.txt` file.

### Important deployment limitations

- This project uses a local SQLite sign-in prototype and writes accepted/rejected sample JSON files to the app filesystem. On Community Cloud, local files are not a durable database and can be lost when the app is restarted or rebuilt. The sign-in prototype is not production authentication. Do not collect real passwords or sensitive personal information from public users. For a public production service, replace local authentication and local sample storage with a managed identity provider and persistent database/object storage.
- The trained `.keras` model is included in the Git repository for this small demo. Do not commit user data, `data/users.sqlite`, secrets, or accepted/rejected sample records.
- Streamlit Community Cloud supports Python versions that are still receiving security updates; choose the same Python version used to test the app. See the official deployment guide for the current list and deployment controls.
