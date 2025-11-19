# GLiNER-BioMed

<p align="center">
  <img src="gliner_biomed.png" alt="GLiNER-BioMed logo">
</p>

---

## 🔬 Overview

GLiNER-BioMed introduces a specialized suite of efficient open biomedical Named Entity Recognition (NER) models based on the GLiNER framework. GLiNER-BioMed leverages synthetic annotations distilled from large generative biomedical language models to achieve state-of-the-art zero-shot and few-shot performance in biomedical entity recognition tasks.

## 🚀 Publicly available models

We publicly released pre-trained GLiNER-BioMed models at multiple scales and variants (uni-encoder and bi-encoder). You can directly access and use these models from Hugging Face Hub:

| Model type | Uni-encoder model | Bi-encoder model |
|------------|-------------------|------------------|
| **Small**  | [gliner-biomed-small-v1.0](https://huggingface.co/Ihor/gliner-biomed-small-v1.0) | [gliner-biomed-bi-small-v1.0](https://huggingface.co/Ihor/gliner-biomed-bi-small-v1.0) |
| **Base**   | [gliner-biomed-base-v1.0](https://huggingface.co/Ihor/gliner-biomed-base-v1.0) | [gliner-biomed-bi-base-v1.0](https://huggingface.co/Ihor/gliner-biomed-bi-base-v1.0) |
| **Large**  | [gliner-biomed-large-v1.0](https://huggingface.co/Ihor/gliner-biomed-large-v1.0) | [gliner-biomed-bi-large-v1.0](https://huggingface.co/Ihor/gliner-biomed-bi-large-v1.0) |

## 📚 Released corpora

To support reproducibility and further research, we also release the corpora used in pre- and post-training stages:

- [Pre-training corpus](https://huggingface.co/datasets/anthonyyazdaniml/gliner-biomed-pre-training)
- [Post-training corpus](https://huggingface.co/datasets/anthonyyazdaniml/gliner-biomed-post-training)

## 🧩 Synthetic annotation model

We also release the synthetic annotation model used to generate large-scale biomedical entity annotations:

- [OpenBioLLM-Text2Graph-8B](https://huggingface.co/Ihor/OpenBioLLM-Text2Graph-8B)

---

## 📦 Installation & usage

### Installation

Install the official GLiNER library with pip:
```bash
pip install gliner
```

### Usage example

After installing the GLiNER library, you can easily load a GLiNER-BioMed model and perform named entity recognition:

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("Ihor/gliner-biomed-large-v1.0")

text = """
The patient, a 45-year-old male, was diagnosed with type 2 diabetes mellitus and hypertension.
He was prescribed Metformin 500mg twice daily and Lisinopril 10mg once daily. 
A recent lab test showed elevated HbA1c levels at 8.2%.
"""

labels = ["Disease", "Drug", "Drug dosage", "Drug frequency", "Lab test", "Lab test value", "Demographic information"]

entities = model.predict_entities(text, labels, threshold=0.5)

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

If you use GLiNER-BioMed models or resources in your research, please cite our work:

```
@misc{yazdani2025glinerbiomedsuiteefficientmodels,
      title={GLiNER-BioMed: A Suite of Efficient Models for Open Biomedical Named Entity Recognition}, 
      author={Anthony Yazdani and Ihor Stepanov and Douglas Teodoro},
      year={2025},
      eprint={2504.00676},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.00676}, 
}
```

---
