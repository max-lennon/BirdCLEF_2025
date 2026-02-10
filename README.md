# (Under Construction!) Birdsong Classification with Wav2Vec 2.0

## Project Overview

This project implements an end-to-end audio classification pipeline using raw waveform inputs and a pretrained Wav2Vec 2.0 model. The goal is to classify audio clips from wild birds into discrete categories based on their acoustic content.

## Methodology

### 1. Dataset Structure & Audio Loading

The dataset is assumed to follow a standard directory layout:

```
data_root/
├── class_1/
│   ├── sample1.wav
│   ├── sample2.wav
├── class_2/
│   ├── sample3.wav
│   └── ...
```

Each subfolder corresponds to a class label. Audio files are automatically discovered and indexed using supported formats (`.wav`, `.flac`, `.mp3`, `.ogg`, `.m4a`).

A custom `FolderAudioDataset` handles:

* Recursive file discovery,
* Label indexing,
* On-the-fly audio loading via `torchaudio`.

---

### 2. Audio Preprocessing

The data pipeline includes:

* Resampling to a fixed 16 kHz rate,
* Conversion to mono channel format,
* Padding or trimming to a fixed length (4 seconds)

---

### 3. Model Architecture

The model is built on top of **Wav2Vec 2.0 Base**, a self-supervised transformer trained on large-scale speech data.

#### Backbone

* Loaded from `torchaudio.pipelines.WAV2VEC2_BASE`,
* Used as a **frozen feature extractor** by default,
* Outputs frame-level latent representations.

#### Classification Head

A lightweight head is added on top of the backbone:

* Layer normalization,
* Linear projection to class logits,
* Mean pooling across the temporal dimension.

---

### 4. Training & Evaluation

The dataset is randomly split into:

* **80% training**
* **20% validation**

Training uses:

* Cross-entropy loss,
* AdamW optimizer,
* GPU acceleration when available,
* Batched data loading with multiprocessing.

Performance is tracked using:

* Average loss per epoch,
* Classification accuracy on both training and validation sets.

Separate training and evaluation loops ensure correct gradient handling and reproducible metrics.

---

## Results and Insights

### Key Observations

* Pretrained Wav2Vec 2.0 features enable strong performance even with limited training data.
* Freezing the backbone significantly reduces training time and overfitting risk.
* Fixed-length waveform normalization simplifies batching without manual feature engineering.
* Mean temporal pooling provides a strong baseline without complex sequence modeling.
