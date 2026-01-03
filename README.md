<div align="center">

# AI Decoded

**Demystifying the black box.  
Master LLMs and agentic workflows using open-source tools and local models.**

<br>

[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=jupyter)](https://jupyter.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](../../pulls)
[![Issues Welcome](https://img.shields.io/badge/Issues-welcome-blue.svg)](../../issues)
[![Stars](https://img.shields.io/github/stars/AIxorDie/ai-decoded?style=social)](https://github.com/AIxorDie/ai-decoded/stargazers)
[![YouTube Channel](https://img.shields.io/badge/YouTube-AI%20Decoded-FF0000?logo=youtube&logoColor=white)](https://www.youtube.com/@asimmunawar/playlists)


<br>

**YouTube Channel — AI Decoded and More**

<p>
  <a href="https://www.youtube.com/@asimmunawar">
    <img src="https://img.shields.io/badge/Watch%20Now-FF0000?logo=youtube&logoColor=white&style=for-the-badge">
  </a>
</p>

</div>

## 📚 AI Decoded Playlists

To make it easier to follow along, I’ve organized the videos into focused playlists:

<div align="center">

| Playlist | Description |
|----------|-------------|
| **[AI Decoded — All Talks](https://www.youtube.com/watch?v=tymeMBcJVMI&list=PLt-aI0fLbqH8vhKcDQehV1pCRuJxCHevE)** | Every video in the AI Decoded series — start here if you want the full journey. |
| **[LLM Basics](https://www.youtube.com/watch?v=Cm_qmhSEFgs&list=PLt-aI0fLbqH_ci_6z7QAkUzH41yjqTh6u)** | Foundations & core concepts: tokenization, attention, sampling, KV-cache & more. |
| **[LLM Training](https://www.youtube.com/watch?v=mOmtIjYkAIk&list=PLt-aI0fLbqH829sc4T7w8uPAKd1r0E7Ex)** | How models learn: SFT, reward models, RLHF/GRPO, PPO & real training workflows. |
| **[Neuro-Symbolic AI](https://www.youtube.com/watch?v=cVRNIRZ5K24&list=PLt-aI0fLbqH8_uc5YjzoVPRkbBpooMfXt)** | Where logic meets learning — reasoning, structure, knowledge graphs & hybrid AI. |
| **Agents & Tools (Coming Soon)** | Agent workflows, tool-calling, orchestration, planning & autonomous systems. |

</div>


---

## What is AI Decoded?

**AI Decoded** is a **hands-on lab** for learning modern AI systems:

- From **“what is a token?”** to **full Transformer decoders**
- From **toy sampling demos** to **running a real LLM locally**
- From **“hello, reward models”** to **RLHF & GRPO with real code**
- From **simple tools** to **agentic workflows and tool-calling**

The goal:  
> *Understand what’s happening under the hood – then break it, fix it, and extend it yourself.*

---

## Repo Overview

```text
ai-decoded/
├─ src/                  # Main notebooks & code
│  ├─ llm_concepts/      # Core LLM / Transformer concepts
│  ├─ llm_training/      # Reward models, RLHF, GRPO, PPO, etc.
│  ├─ neuro_symbolic_ai/ # All about Neuro-Symbolic AI.
│  ├─ agents/            # Tool-calling, agents & workflows (planned/expanding)
│
├─ docs/                 # Slides, diagrams, and reference material
├─ helping_materials/    # Extra notes, images, helper scripts
├─ LICENSE
└─ README.md
```

---

## Getting Started

### Prerequisites
- Python 3.10 or newer
- Terminal / Command Prompt

```bash
python3 --version
```

### Clone Repo
```bash
git clone https://github.com/AIxorDie/ai-decoded.git
cd ai-decoded
```

### Create Virtual Environment and Install Required Packages
```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install \
  torch \
  datasets \
  jupyter \
  transformers \
  notebook \
  ipywidgets \
  tokenizers \
  accelerate \
  safetensors \
  huggingface-hub \
  matplotlib \
  ollama \
  trl \
  numpy

# Optional
jupyter lab
```

### For Setting up the Full IDE and Coding Assistant(s)
https://github.com/AIxorDie/ai-decoded/blob/main/docs/ide_setup.md

---

## License
MIT License.

