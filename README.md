# GLiNER-biomed
**Repository Under Construction**

---

## 🔬 Overview

GLiNER-biomed introduces a specialized suite of efficient open biomedical Named Entity Recognition (NER) models based on the GLiNER framework. GLiNER-biomed leverages synthetic annotations distilled from large generative biomedical language models to achieve state-of-the-art zero-shot and few-shot performance in biomedical entity recognition tasks.

This repository is currently under development. Complete resources, including data, training pipelines, and further documentation, will soon be available here.

## 🚀 Publicly Available Models

We publicly released pre-trained GLiNER-biomed models at multiple scales and variants (uni-encoder and bi-encoder). You can directly access and use these models from Hugging Face Hub:

| Model Type | Uni-encoder Model | Bi-encoder Model |
|------------|-------------------|------------------|
| **Small**  | [gliner-biomed-small-v1.0](https://huggingface.co/Ihor/gliner-biomed-small-v1.0) | [gliner-biomed-bi-small-v1.0](https://huggingface.co/Ihor/gliner-biomed-bi-small-v1.0) |
| **Base**   | [gliner-biomed-base-v1.0](https://huggingface.co/Ihor/gliner-biomed-base-v1.0) | [gliner-biomed-bi-base-v1.0](https://huggingface.co/Ihor/gliner-biomed-bi-base-v1.0) |
| **Large**  | [gliner-biomed-large-v1.0](https://huggingface.co/Ihor/gliner-biomed-large-v1.0) | [gliner-biomed-bi-large-v1.0](https://huggingface.co/Ihor/gliner-biomed-bi-large-v1.0) |

---

## 📦 Installation & Usage

### Installation

Install the official GLiNER library with pip:
```bash
pip install gliner
```

### Usage Example (Biomedical Domain)

After installing the GLiNER library, you can easily load a GLiNER-biomed model and perform named entity recognition:

```python
from gliner import GLiNER

# Initialize GLiNER with a biomedical model (large variant example)
model = GLiNER.from_pretrained("Ihor/gliner-biomed-large-v1.0")

text = """
The patient, a 45-year-old male, was diagnosed with type 2 diabetes mellitus and hypertension.
He was prescribed Metformin 500mg twice daily and Lisinopril 10mg once daily. 
A recent lab test showed elevated HbA1c levels at 8.2%.
"""

# Biomedical labels for entity prediction
labels = ["Disease", "Drug", "Drug dosage", "Drug frequency", "Lab test", "Lab test value", "Demographic information"]

# Perform entity prediction
entities = model.predict_entities(text, labels, threshold=0.5)

# Display predicted entities and their labels
for entity in entities:
    print(entity["text"], "=>", entity["label"])
```

**Expected output:**
```
45-year-old male => Demographic information
type 2 diabetes mellitus => Disease
hypertension => Disease
Metformin => Drug
500mg => Drug dosage
twice daily => Drug frequency
Lisinopril => Drug
10mg => Drug dosage
once daily => Drug frequency
HbA1c levels => Lab test
8.2% => Lab test value
```

For more detailed documentation and usage examples, visit the official [GLiNER repository](https://github.com/urchade/GLiNER).

---

## 📌 Citation

If you use GLiNER-biomed models or resources in your research, please cite our work:

```
Citation not available yet.
```

---

## 🛠️ Repository Status

⚠️ **This repository is currently under construction.**  
Updates, full documentation, and complete pipelines will soon be available.

---

🌟 **Stay tuned!**
