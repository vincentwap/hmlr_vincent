
# BBC News NLP Pipeline

This project demonstrates an end-to-end NLP pipeline using transformer models for BBC news and sport articles. It handles:

- Convert zip from BBC dataset available from http://mlg.ucd.ie/datasets/bbc.html
- Sub-category classification for Business, Entertainment, and Sports using keyword heuristics
- Named Entity Recognition (NER) with spaCy to identify media personalities and their roles
- Visualisation of results using seaborn and matplotlib
- Transformers for Classification - Use DistilBERT via Hugging Face to fine-tune on BBC article subcategories.
- Event summary extraction for anything scheduled or that occurred in April
- Large Language Models (LLMs) and their fine-tuning

did not cover below

- Gender bias analysis
- Text augmentation using `nlpaug`
- Fine-tuning DistilBERT on custom labels

---

## Project Structure

```
bbc-news-nlp-pipeline/
│
├── data/
│   ├── bbc_news_dataset.csv
│   ├── bbc_sport_dataset.csv
│   ├── bbc-fulltext.zip
│   └── bbcsport-fulltext.zip
│
├── notebooks/
│   └── main.ipynb               # Core development notebook
│
├── scripts/
│   └── fine_tune_distilbert.py  # Fine-tuning script
│
└── models/                      # Fine-tuned models will be saved here
```

---

## System Requirements

This project was developed and tested on the following configuration:

- **Laptop RAM**: 8GB
- **GPU**: Optional (Recommended: 4GB+ VRAM for fine-tuning)
- **Processor**: Intel i5
- **Disk**: 5GB free
- **Environment**: Jupyter Notebook (Anaconda)

---

## Installation

```bash
pip install transformers datasets nlpaug scikit-learn spacy
python -m spacy download en_core_web_sm
```

---

## How to Run

1. Launch Jupyter Notebook
2. Open `notebooks/main.ipynb`
3. Run the cells step-by-step:
   - Extract ZIP files
   - Preprocess text
   - Run zero-shot classification
   - Perform NER with job tagging
   - Summarise April events
   - Run gender bias audit
   - Fine-tune DistilBERT with script or interactively

---

## Fine-Tuning (Alternative Script-Based)

Use the standalone training script:
```bash
cd scripts
python fine_tune_distilbert.py
```

---

## Output

- `bbc_enriched_dataset.csv`: NLP enriched dataset
- `bbc_april_summaries.csv`: Summarised April content
- `models/`: Trained model and tokenizer

---

## License

MIT License

---

## Contact

Vincent Nwaka
- vincentwap@gmail.com
- https://vincentwap.github.io/vincent/

## Challenges & System Limitation

While working on this project, I attempted to run advanced NLP models, I encountered significant limitations due to my local system configuration.

To address this, I switched to a lighter transformer — DistilBERT, a distilled version of BERT — for the fine-tuning phase. DistilBERT consumes significantly fewer resources while retaining much of the performance of larger models. It fine-tuned successfully and performed well for my classification task.
