# Deep Learning Beleg - Gruppe 1

This repository contains our Deep Learning model for classifying machine health using sequential acceleration data.

## Team Setup Instructions

**1. Get the Data**
* Download `Datensatz.csv` from OPAL.
* Place it in the main folder of this project (Do not commit this to GitHub! File is too large to be commited).

**2. Install the Package Manager (`uv`)**
* **Windows:** Open PowerShell and run: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
* **macOS/Linux:** Open Terminal and run: `curl -LsSf https://astral.sh/uv/install.sh | sh`

**3. Build the Virtual Environment**
Open your terminal inside this project folder and run:
* `uv venv`

**4. Install the Dependencies**
Run this command to automatically install all required libraries (including PyTorch, Jupyter, and Optuna):
* `uv pip install -r requirements.txt`

**5. Activate the Environment**
Open the Jupyter Notebook in VS Code, click "Select Kernel" in the top right, choose the new `.venv` folder, and the code *should* be ready to run. Unless errors pop up... good luck! `:)`