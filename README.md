# Hindi Tacotron2: End-to-End Hindi Text-to-Speech Synthesis

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<br />
<div align="center">
  <h3 align="center">Hindi Tacotron2 Text-to-Speech</h3>

  <p align="center">
    An end-to-end Hindi Text-to-Speech (TTS) system built using the Tacotron2 architecture. The model converts Hindi text into mel spectrograms using a custom Hindi Akshar tokenizer and reconstructs speech using the Griffin-Lim algorithm.
    <br />
    <a href="https://huggingface.co/spaces/DYNAMAXD/Tacotron2_Hindi"><strong>Try the Live Demo »</strong></a>
    <br />
    <br />
    <a href="https://huggingface.co/DYNAMAXD">Hugging Face</a>
    ·
    <a href="https://github.com/DYNAMAXD/Tacotron2-Implementation/issues">Report Bug</a>
    ·
    <a href="https://github.com/DYNAMAXD/Tacotron2-Implementation/issues">Request Feature</a>
  </p>
</div>

---

## Overview

This repository contains a complete implementation of a Hindi Text-to-Speech pipeline based on the Tacotron2 architecture.

The project includes:

- Custom Hindi Akshar Tokenizer
- Complete Tacotron2 implementation in PyTorch
- Griffin-Lim waveform reconstruction
- Interactive Gradio Web Interface
- Hugging Face Spaces deployment
- Training and inference scripts
- Attention visualization
- Mel spectrogram visualization

The model has been trained on the IISc SYSPIN Hindi Speech Corpus and is capable of synthesizing intelligible Hindi speech directly from text.

---

## Features

- End-to-End Hindi Text-to-Speech
- Tacotron2 Encoder-Decoder Architecture
- Custom Hindi Akshar Tokenizer
- Griffin-Lim Audio Reconstruction
- Mel Spectrogram Generation
- Attention Alignment Visualization
- Gradio Web Interface
- Hugging Face ZeroGPU Deployment
- PyTorch Implementation

---

## Model Pipeline

```
Hindi Text
      │
      ▼
Hindi Akshar Tokenizer
      │
      ▼
Tacotron2 Encoder
      │
      ▼
Location Sensitive Attention
      │
      ▼
Tacotron2 Decoder
      │
      ▼
Postnet
      │
      ▼
Mel Spectrogram
      │
      ▼
Griffin-Lim Reconstruction
      │
      ▼
Generated Speech (.wav)
```

---

## Dataset

The model was trained using the **IISc SYSPIN Hindi Speech Corpus** [click here][dataset-link].

| Property | Value |
|----------|-------|
| Language | Hindi |
| Speaker | Male |
| Sample Rate | 22050 Hz |
| Type | Single Speaker, Professionally Recorded |

---

## Built With

* ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
* ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
* ![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge)
* ![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)
* ![Librosa](https://img.shields.io/badge/Librosa-Audio-blue?style=for-the-badge)

---

## Repository Structure

```text
Tacotron2-Implementation
│
├── app.py
├── inference.py
├── train.py
├── prep_data.py
├── dataset.py
├── tokenizer_hindi.py
├── model.py
├── config.py
├── requirements.txt
├── vocab.pt
├── checkpoint_0035.pt
│
├── audio_out/
├── inference_out/
└── README.md
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/DYNAMAXD/Tacotron2-Implementation.git
cd Tacotron2-Implementation
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## How to Use

### Step 1 - Configure paths and settings

Open `config.py` and update the following before doing anything else:

- `WAV_DIR` - path to the folder containing your `.wav` files
- `JSON_PATH` - path to your transcripts file (an Excel sheet with WAV filenames and their corresponding Hindi transcripts)
- `NUM_SAMPLES` - number of samples you want to train on
- `NUM_EPOCHS` - number of training epochs

The transcript file is expected to be an Excel sheet where each row has the WAV filename and its corresponding transcript text.

### Step 2 - Prepare vocabulary and data splits

Run `prep_data.py` to build the vocabulary file and create the train/val CSV splits:

```bash
python prep_data.py
```

This script does two things:

- Builds `vocab.pt`, a precomputed vocabulary file using the Akshar tokenizer (the vocab building section is commented out by default - uncomment it on first run)
- Creates `train.csv` and `test.csv` which are used during training

After the first run you can comment the vocab building back out so it does not recompute on every run.

If you want to change the architecture - number of layers, attention dimensions, postnet depth, etc. - edit `model.py` before starting training.

### Step 3 - Train

```bash
python train.py
```

Training metrics and alignment plots are saved automatically as training progresses.

### Step 4 - Inference

```bash
python inference.py --text "नमस्ते, आपका स्वागत है।"
```

Or run the Gradio interface:

```bash
python app.py
```

Then open `http://127.0.0.1:7860` in your browser.

---

## Example Inputs

```
नमस्ते, आपका स्वागत है।
भारतीय रेलवे में आपका स्वागत है।
कृपया अपने सामान का ध्यान रखें।
आज का मौसम बहुत सुहावना है।
अगला स्टेशन नई दिल्ली है।
```

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | Tacotron2 |
| Framework | PyTorch |
| Sample Rate | 22050 Hz |
| Mel Channels | 80 |
| FFT Size | 1024 |
| Hop Length | 256 |
| Vocabulary Size | 4685 |
| Griffin-Lim Iterations | 60 |
| Batch Size | 24 |
| Epochs | 36 |

---

## Deployment

The project is deployed using:

- Hugging Face Spaces
- Gradio
- ZeroGPU Runtime

---

## Future Work

- HiFi-GAN Integration
- Multi-Speaker Hindi TTS
- Emotion-aware Speech Synthesis
- ONNX Export
- Faster Real-time Inference
- Streaming TTS
- Speaker Adaptation

---

## Acknowledgements

A special thanks to **Priyam Mazumdar** for creating one of the most comprehensive Tacotron2 tutorial series and for openly sharing the implementation that served as an excellent educational reference during the development of this project. The insights from the tutorial greatly helped in understanding the Tacotron2 architecture and training pipeline.

GitHub: https://github.com/priyammaz

This repository extends and adapts those concepts for Hindi Text-to-Speech, including a custom Hindi tokenizer, data preprocessing pipeline, training workflow, inference utilities, and Hugging Face deployment.

---

## Author

**Divyansh Yadav**

GitHub: https://github.com/DYNAMAXD

LinkedIn: https://www.linkedin.com/in/dynamaxd/

---

## License

This project is intended for research and educational purposes.

If you use this work in your research, please consider citing this repository and the original Tacotron2 paper. The Tacotron family of models provides an end-to-end sequence-to-sequence architecture for neural text-to-speech synthesis.

---

<!-- MARKDOWN LINKS -->

[contributors-shield]: https://img.shields.io/github/contributors/DYNAMAXD/Tacotron2-Implementation.svg?style=for-the-badge
[contributors-url]: https://github.com/DYNAMAXD/Tacotron2-Implementation/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/DYNAMAXD/Tacotron2-Implementation.svg?style=for-the-badge
[forks-url]: https://github.com/DYNAMAXD/Tacotron2-Implementation/network/members
[stars-shield]: https://img.shields.io/github/stars/DYNAMAXD/Tacotron2-Implementation.svg?style=for-the-badge
[stars-url]: https://github.com/DYNAMAXD/Tacotron2-Implementation/stargazers
[issues-shield]: https://img.shields.io/github/issues/DYNAMAXD/Tacotron2-Implementation.svg?style=for-the-badge
[issues-url]: https://github.com/DYNAMAXD/Tacotron2-Implementation/issues
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/dynamaxd/
[dataset-link]:https://syspin.iisc.ac.in/datasets
