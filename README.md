## Detect AI vs. Human-Generated Images 🖼️🤖

Lightweight Flask service and supporting notebooks for distinguishing AI-generated imagery from real-world photographs.

A concise project for experimenting with model training and inference: includes training notebooks, exported Keras models, and a minimal web UI to upload an image and get a prediction (AI-generated vs. real). Intended for research, demos, and educational use — not a certified forensic tool.

### Example run (captured outputs) 📈

Train CSV sample:

```
    Unnamed: 0                                        file_name  label
0           0  train_data/a6dcb93f596a43249135678dfcfc17ea.jpg      1
1           1  train_data/041be3153810433ab146bc97d5af505c.jpg      0
2           2  train_data/615df26ce9494e5db2f70e57ce7a3a4f.jpg      1
3           3  train_data/8542fe161d9147be8e835e50c0de39cd.jpg      0
4           4  train_data/5d81fa12bc3b4cea8c94a6700a477cf2.jpg      1
```

Sample image path:

```
/kaggle/input/ai-vs-human-generated-dataset/train_data/a6dcb93f596a43249135678dfcfc17ea.jpg
File exists: True
```

Dataset split & generators:

```
Training samples: 63960, Validation samples: 15990
Found 63960 validated image filenames belonging to 2 classes.
Found 15990 validated image filenames belonging to 2 classes.
```

Model summary (top / params):

```
Total params: 23,589,761
Trainable params: 2,049
Non-trainable params: 23,587,712
```

Epoch-by-epoch metrics (10 epochs):
| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|-------|-----------|------------|---------|----------|
| 1 | 0.6237 | 0.6559 | 0.7813 | 0.4844 |
| 2 | 0.7581 | 0.5104 | 0.7827 | 0.4706 |
| 3 | 0.7649 | 0.4973 | 0.7873 | 0.4626 |
| 4 | 0.7654 | 0.4961 | 0.7919 | 0.4526 |
| 5 | 0.7707 | 0.4875 | 0.7934 | 0.4496 |
| 6 | 0.7742 | 0.4846 | 0.7929 | 0.4563 |
| 7 | 0.7713 | 0.4819 | 0.7964 | 0.4462 |
| 8 | 0.7755 | 0.4799 | 0.7954 | 0.4503 |
| 9 | 0.7707 | 0.4815 | 0.7978 | 0.4442 |
| 10 | 0.7800 | 0.4738 | 0.8004 | 0.4406 |

Saved model ✅:

```
Model saved as resnet50_ai_vs_real_final.h5
```

Example inference:

```
Prediction: Real, Probability: 0.1724
```

Repository layout 📁:

```
├── docs/                     # GitHub Pages documentation site
├── models/                   # Exported Keras models used for inference
├── notebooks/                # Experiment and training notebooks
├── src/
│   └── ai_vs_human_detector/
│       ├── app.py            # Flask application factory and routes
│       └── templates/        # Jinja templates served by Flask
├── uploads/                  # Temporary user uploads (ignored by git)
├── .gitignore
├── README.md
└── requirements.txt
```

### 1. Clone the repository 📥

```powershell
git clone https://github.com/cepdnaclk/e20-co542-Detect-AI-vs-Human-Generated-Images.git
cd e20-co542-Detect-AI-vs-Human-Generated-Images
```

### 2. (Optional) Create and activate a virtual environment 🐍

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Command Prompt:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies ⬇️

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Prepare a trained model 🧠

- If you can train locally with a GPU:

  1. Open a notebook from `notebooks/` (e.g., `resnet50.ipynb`).
  2. Train using the dataset.
  3. Export the trained `.h5` file into the `models/` directory.

- If local training is not feasible:
  1. Use Google Colab or Kaggle.
  2. Use the competition dataset: https://www.kaggle.com/competitions/detect-ai-vs-human-generated-images
  3. Download the trained weights and copy them into `models/`.

Name the exported file to match a candidate expected by the app (for example `models/resnet50_ai_vs_real_best.h5`).

If you prefer a ready-made model, place the downloaded file in `models/`. (https://drive.google.com/file/d/1IlMIVhLpmD2G36HWOWy2nN510xGyRkDu/view?usp=sharing)

### 5. Run the web service ▶️

Below are a few equivalent ways to start the Flask app on different shells. Pick the style you prefer. In all cases make sure your virtual environment is activated first (see section 2).

PowerShell (set env for current session):

```powershell
$env:FLASK_APP = 'src.ai_vs_human_detector.app:app'
python -m flask run --debug
```

Command Prompt (set env for current session):

```cmd
set FLASK_APP=src.ai_vs_human_detector.app:app
python -m flask run --debug
```

Modern (recommended) Flask CLI — avoid environment variables and pass the app directly to the CLI.

PowerShell / Command Prompt / POSIX (works when `flask` is on PATH or inside an activated venv):

```powershell
flask --app src.ai_vs_human_detector.app --debug run
```

Or, POSIX-style environment variable inline (macOS / Linux):

```bash
FLASK_APP=src.ai_vs_human_detector.app:app python -m flask run --debug
```

Notes 💡

- Use the virtual environment activation commands from section 2 before running the above (so the `flask` command and project deps are available).
- `--debug` enables the debug mode (reloader and debugger). If your Flask version doesn't accept `--debug`, you can instead run with `--reload` or set `FLASK_ENV=development` on older Flask versions.
- If you see an import or runtime error after starting Flask, double-check that `models/` contains a compatible model file and that your Python environment has the packages listed in `requirements.txt` installed.

When Flask starts it prints a http://127.0.0.1:5000/ link—open it in a browser, upload an image, and the endpoint will respond with either "AI-generated" or "Real" plus a confidence score. 🚀

Notes

- Ensure a compatible model file exists in `models/`. The app scans `models/` for the first filename matching entries in MODEL_CANDIDATES.
- Update the Google Drive placeholder link to a real download URL if you reference a prebuilt model.
- Use matching environment activation and FLASK_APP export commands for your shell; instructions above provide PowerShell, CMD, and POSIX variants.
