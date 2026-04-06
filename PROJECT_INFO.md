# 🛡️ CRIMSON-India: In-Depth Technical Documentation

**CRIMSON-India** (Crime Real-time Intelligence Monitoring & Safety online News) is a state-of-the-art Deep Learning framework designed to monitor, categorize, and verify public safety incidents from Indian news media in real-time.

---

## 🏗️ Core Architecture: The E2E Data Pipeline

The system is built as a high-performance modular pipeline, ensuring scalability from local scrapers to large-scale multilingual classification.

### 1. Ingestion Engine
- **RSS Streams (`scraper/rss_scraper.py`)**: Consumes live XML feeds from major Indian outlets (TOI, NDTV, The Hindu).
- **Universal Scraper (`scraper/news_scraper.py`)**: A Selenium/BS4 fallback for deep-web article extraction.
- **Simulation Layer (`utils/data_annotator.py`)**: Generates statistically accurate synthetic news clusters for training and benchmarking.

### 2. Linguistic Intelligence Layer
- **Standardization (`preprocessing/text_cleaner.py`)**: Implements language-aware cleaning, stripping web noise while retaining Devanagari and regional scripts.
- **Language Routing**: Uses `langdetect` to intelligently route content to the most effective AI model.

### 3. Progressive AI Classification Hub
The framework utilizes a multi-tier model hierarchy for maximum flexibility:

| Tier | Category | Model Architecture | Primary Use Case |
| :--- | :--- | :--- | :--- |
| **L1** | **Baseline** | TF-IDF + Linear SVM | Real-time dashboard inference with zero latency. |
| **L2** | **Sequential** | PyTorch BiLSTM + FastText | Captures local context and word ordering. |
| **L3** | **Global** | XLM-RoBERTa (Base) | High-accuracy multilingual/English classification. |
| **L4** | **Localized** | MuRIL (Google) | Optimized for Indian linguistic nuances and code-switching. |

### 4. Geospatial & Verification Modules
- **Dynamic Geocoding (`utils/geocoder.py`)**: Employs regex-based NER to identify city entities and map them to precise GPS coordinates across India.
- **Verification Hub (`utils/fact_checker.py`)**: Uses keyword overlap and cross-feed searching to verify news authenticity.

---

## 🛠️ Full Technical Stack

### **Data & Infrastructure**
- **Python 3.10+**
- **Pandas / Numpy**: Vectorized data operations.
- **Feedparser**: XML parsing logic.
- **Scipy**: Statistical Pearson-r correlation tests.

### **Deep Learning Hub**
- **PyTorch**: Backend for BiLSTM and Transformer layers.
- **Hugging Face (`transformers`)**: Infrastructure for MuRIL & XLM-R.
- **Scikit-Learn**: Multi-label SVM strategies.

### **Visualization & Frontend**
- **Streamlit**: Industrial-grade interactive dashboard.
- **Matplotlib / Seaborn / Altair**: Advanced data visualization hooks.

---

## 📂 Project Structure Walkthrough

- `app.py`: The primary Streamlit entry point.
- `main.py`: The executable pipeline for training and batch processing.
- `models/`: High-level implementations of all classification architectures.
- `analysis/`: Logic for trend generation and NCRB (National Crime Records Bureau) stats correlation.
- `data/`: Local storage for datasets and processed statistics.
- `utils/`: Core utilities for geocoding, fact-checking, and logging.

---
**CRIMSON-India** is designed for researchers and public safety analysts to gain a data-driven overview of the safety landscape in India.
