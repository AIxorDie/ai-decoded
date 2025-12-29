# Setting up the IDE with Coding Assistant

This guide walks you through installing **Visual Studio Code**, setting up **Ollama**, and configuring the **Continue** extension to use the **gpt-oss-20b** model as a local coding assistant. _You can use any other model that you prefer_.

---

## Prerequisites

- A computer running **Windows, macOS, or Linux**
- Internet access for downloads
- At least **16 GB RAM** recommended for 20B models (more is better)
- 40+ GB of free disk space
- (Optional) A GPU with over **12 GB VRAM** minimum can significantly enhance inference speeds

---

## 1. Install Visual Studio Code

1. Go to the official VS Code website:  
   https://code.visualstudio.com
2. Download the installer for your operating system.
3. Run the installer and follow the on-screen instructions.
4. Launch **Visual Studio Code** once installation is complete.


---

## 2. Install GitHub Copilot Extension

GitHub Copilot offers a limited free tier (Copilot Free) with monthly allowances for code completions and chat, plus free access for verified students, teachers. The free plan is great for moderate users.

### Install the Continue Extension in VS Code

1. Open Visual Studio Code
- Click the Extensions icon on the left sidebar
- Search for **GitHub Copilot**
- Install the extension
- Reload VS Code

Copilot gives you suggestions as you type, you can also interact with it by

- **Command+Shift+I** on Mac
- **Ctrl+Alt+I** on Windows

---

## 3. Install Continue.Dev with Local Models

Ollama lets you run large language models locally.

### Install Ollama

1. Go to:  
   https://ollama.com
2. Download the installer for your operating system.
3. Install Ollama using the default settings.
4. After installation, open a terminal (Command Prompt, PowerShell, or Terminal).

### Verify Installation

In the terminal, run:
```bash
ollama --version
```
If a version number appears, Ollama is installed correctly.


### Download the gpt-oss-20b Model

In your terminal, run:
```bash
ollama pull gpt-oss:20b
ollama pull qwen2.5-coder-32b
```
This may take some time depending on your internet speed.

To verify the model is available:
```bash
ollama list
```
You should see **gpt-oss:20b** in the list. You can try to see if the model is running properly:

```bash
ollama run gpt-oss:20b "What is the capital of Japan?"
```
If you get a response, you have the model downloaded properly.

_Note: If you are not able to run cannot run **gpt-oss-20b** model. Try with a much smaller model like **Llama 3.1 8B**_

### Install the Continue Extension in VS Code

1. Open Visual Studio Code
- Click the Extensions icon on the left sidebar
- Search for **Continue**
- Install the extension named “Continue – AI Code Assistant”
- Reload VS Code if prompted

### Configure Continue to use the model

Continue does a good job of making the config very easy via GUI. Click on the **Continue** extension, select Ollama, and the correct model. The _~/.continue/config.yaml_ should now look something like. 

```yaml
name: Local Config
version: 1.0.0
schema: v1
models:
  - name: gpt-oss-20b
    provider: ollama
    model: gpt-oss:20b
```

By default, the extension will select **openai/gpt-oss-20b** as the model, but that is wrong identifier for ollama and must be changed to **gpt-oss:20b**. You can always confirm the exact name by running _ollama list_.

You can enable code Continue code assistant by **Command+I** on Mac.

---

## 4. Python Virtual Environment Setup

### Prerequisites
- Python 3.10 or newer
- Terminal / Command Prompt

```bash
python3 --version
```

### Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Install Required Packages

```bash
pip install --upgrade pip

pip install \
  torch \
  transformers \
  notebook \
  ipywidgets \
  tokenizers \
  accelerate \
  safetensors \
  huggingface-hub \
  ollama \
  numpy
```