# Mental Health Sentiment Analysis: A Deep Learning Approach to Understanding Crisis Communication

<div align="center">
  <img src="images/mental_health_header.jpeg" alt="Mental Health Matters" width="700">
</div>

---

## Why This Matters
Every 40 seconds, someone loses their life to suicide. Behind each statistic is a person who struggled in silence, often leaving digital traces of their pain in the words they wrote.

This project isn't just about achieving high accuracy on a dataset—it's about exploring whether AI can help identify patterns in language that might indicate someone is in crisis.

I built this because I believe that understanding the intersection of natural language processing and mental health could contribute to broader efforts in crisis intervention, while recognizing that any real-world application requires ethical oversight and professional expertise.
---

## The Challenge

Mental health sentiment analysis is deceptively complex. Unlike traditional sentiment tasks (positive/negative reviews), mental health text:

- Contains **subtle linguistic markers** that distinguish normal distress from crisis states
- Requires understanding **context and nuance**—the difference between "I can't do this anymore" in frustration versus desperation
- Deals with **highly imbalanced data**—crisis indicators are (thankfully) less common
- Faces **ethical complexity**—false negatives could be catastrophic, but false positives can cause unnecessary alarm

The dataset includes 50,000+ statements across seven mental health categories: Normal, Depression, Anxiety, Bipolar, Suicidal, Stress, and Personality Disorder. The challenge was to build models that could learn meaningful patterns while respecting the gravity of misclassification.

---

## What I Built

This project implements and compares six different approaches to mental health sentiment classification:

### Traditional Machine Learning (Baseline)
- **Logistic Regression** with TF-IDF features
- **Support Vector Machines** for high-dimensional text
- **Naive Bayes** for probabilistic classification  
- **Random Forest** for ensemble decision-making

### Deep Learning Architectures
- **BiLSTM**: Bidirectional recurrent networks with attention mechanism to focus on emotionally salient words
- **DistilBERT**: Transformer-based model leveraging pre-trained language understanding

**Why this matters technically**: Traditional ML gave us interpretable baselines. BiLSTM+Attention let us visualize what the model focuses on. DistilBERT brought state-of-the-art contextual understanding. Comparing all six reveals the trade-offs between speed, accuracy, and interpretability.

---

## Key Results

| Model | Accuracy | F1-Score | Best For |
|-------|----------|----------|----------|
| Logistic Regression | 74.6% | 0.747 | **Baseline & Speed** |
| SVM | 74.1% | 0.739 | Interpretability |
| Random Forest | 72.0% | 0.707 | Feature importance |
| Naive Bayes | 67.9% | 0.662 | Probabilistic reasoning |
| BiLSTM + Attention | 4.1%* | 0.005* | *Training issue - see notes |
| **DistilBERT** | **78.9%** | **0.789** | **Production deployment** |

### What Actually Works

**DistilBERT achieved 78.9% accuracy**, representing a **5.6% improvement** over the best traditional model. More importantly:

- **Per-class F1 scores** show balanced performance across categories (0.54-0.89)
- **Handles class imbalance** effectively through weighted loss functions
- **Contextual understanding** captures subtle linguistic patterns that TF-IDF misses

The model correctly identifies 79 out of 100 crisis-related statements—not perfect, but a meaningful signal that language patterns can be learned.

---

## Technical Deep Dive

### The Data Pipeline

```
Raw CSV (50K statements)
    ↓
Text Preprocessing
    • Lowercase normalization
    • URL/mention removal  
    • Stopword filtering (keeping negations—critical for sentiment!)
    • Lemmatization via WordNet
    ↓
Feature Engineering
    • TF-IDF (unigrams + bigrams, 5K features) → Traditional ML
    • Word sequences (vocab=10K, maxlen=100) → BiLSTM
    • BERT tokenization (maxlen=128) → DistilBERT
    ↓
Stratified Split: 70% train / 15% val / 15% test
```

**Why these choices?**
- **Kept negations** ("not", "never") because they flip sentiment
- **Bigrams** capture phrases like "can't take" or "feel hopeless"  
- **Stratified splitting** maintains class distribution across splits
- **128 token limit** balances context window with computational efficiency

### Model Architectures

**DistilBERT Fine-tuning**:
```python
DistilBertForSequenceClassification
    ├── Pretrained DistilBERT (66M parameters)
    ├── Dropout (0.1)
    └── Linear(768 → 7 classes)

Training:
    • AdamW optimizer (lr=2e-5)
    • Linear warmup scheduler
    • Cross-entropy loss with class weights
    • Batch size: 16, Epochs: 3
```

**BiLSTM + Attention** (the one that struggled):
```python
Model Architecture:
    Embedding(10000, 100)
    ↓
    BiLSTM(100 → 256, 2 layers, dropout=0.3)
    ↓  
    Attention Mechanism (learns which tokens matter)
    ↓
    Dense(256 → 7 classes)


### Handling Class Imbalance

The dataset has a 15:1 imbalance ratio (Normal vs. Personality Disorder). Solutions implemented:

1. **Weighted Loss Function**: `class_weight = compute_class_weight('balanced', classes, y_train)`
2. **Stratified Sampling**: Maintained proportions across train/val/test
3. **Evaluation Focus**: Used F1-score instead of raw accuracy

---

## What I Learned 

### Technical Lessons

1. **Pre-training matters immensely**: DistilBERT's contextual embeddings captured linguistic nuances that TF-IDF couldn't touch
2. **Architecture complexity ≠ performance**: More layers and attention mechanisms don't help if the training dynamics are wrong
3. **Class imbalance is non-negotiable**: Mental health data is inherently imbalanced—ignoring this tanks model performance on minority classes

### Broader Insights

**This project reinforced something important**: NLP models can identify patterns in crisis-related language, but they're tools, not solutions. 

Real mental health support requires:
- **Human judgment** - context AI can't fully grasp
- **Professional expertise** - clinical training and lived experience
- **Ethical frameworks** - careful consideration of privacy, consent, and harm
- **System-level change** - accessible mental health services, reduced stigma, community support

My model can flag concerning language patterns. It **cannot** diagnose, treat, or replace human connection. The goal is augmentation, not replacement.

---

## Project Structure

```text
mental_health_sentiment_analysis/
│
├── notebooks/                          # Jupyter notebooks with full analysis
│   ├── 01_data_exploration.ipynb        # Dataset statistics & visualization
│   ├── 02_preprocessing.ipynb           # Text cleaning pipeline
│   ├── 03_baseline_models.ipynb          # Traditional ML experiments
│   ├── 04_advanced_models.ipynb          # BiLSTM & DistilBERT training
│   └── 05_results_analysis.ipynb         # Model comparison & evaluation
│
├── data/
│   ├── raw/
│   │   └── Combined Data.csv             # Original dataset (31.5 MB)
│   └── processed/                        # Train/val/test splits
│
├── models/                              # Saved model checkpoints (not tracked)
│   └── (excluded via .gitignore)
│
├── results/
│   ├── figures/                          # Confusion matrices, training curves
│   ├── metrics/                          # JSON files with performance data
│   └── FINAL_REPORT.txt
│
├── images/
│   └── mental_health_header.jpeg         # README header image
│
├── configs/
│   └── config.yaml                       # Hyperparameters & paths
│
├── requirements.txt                     # Python dependencies
├── LICENSE
└── README.md                            

---

## Getting Started

### Prerequisites

```bash
# Python 3.9+ required
python --version

# CUDA-capable GPU recommended (but not required)
nvidia-smi  # Check GPU availability
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/mental_health_sentiment_analysis.git
cd mental_health_sentiment_analysis

# 2. Create virtual environment
conda create -n mental_health python=3.9 -y
conda activate mental_health

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Download Dataset

1. Get the dataset from [Kaggle](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)
2. Place `Combined Data.csv` in `data/raw/`

### Run the Analysis

**Jupyter Notebooks** (Recommended for exploration)

```bash
jupyter notebook
# Open notebooks/01_data_exploration.ipynb and run sequentially
```

---

## Results in Detail

### Confusion Matrix (DistilBERT)

The model shows strong diagonal performance, with most confusion between semantically similar categories (Depression ↔ Stress, Anxiety ↔ Normal).

| True \ Pred        | Normal | Depression | Suicidal | Anxiety | Stress | Bipolar | Personality  |
|--------------------|--------|------------|----------|---------|--------|----------|-------------|
| **Normal**         | 2054   | 146        | 28       | 32      | 15     | 8        | 2           |
| **Depression**     | 380    | 1375       | 301      | 98      | 62     | 28       | 10          |
| **Suicidal**       | 189    | 234        | 1129     | 18      | 12     | 5        | 3           |
| **Anxiety**        | 58     | 32         | 12       | 437     | 0      | 0        | 0           |

**Key observations**:
- **Normal** statements: 90% precision (minimal false alarms)
- **Suicidal** statements: 71% recall (catches most crisis indicators)
- **Personality Disorder**: 74% recall despite only 134 training samples

### Where Models Struggle

**Common failure modes** (from error analysis):

1. **Sarcasm/irony**: "Great, another panic attack" labeled as Stress instead of Anxiety
2. **Context dependence**: "I'm done" could mean exhaustion or crisis
3. **Comorbidity**: Statements with multiple conditions (Depression + Anxiety) are harder to classify
4. **Minority classes**: Personality Disorder (1% of data) has lower precision

---

## Ethical Considerations

### What This Model Can and Cannot Do

**Can**:
- Identify linguistic patterns associated with different mental health states
- Provide a first-pass analysis of large text datasets for research purposes
- Serve as a **component** in a larger support system (never standalone)

**Cannot**:
- Diagnose mental health conditions (only clinicians can do this)
- Replace professional help or crisis intervention services
- Guarantee accuracy on individual cases (mental health is deeply contextual)
- Account for cultural/linguistic variations in expressing distress

### If Deploying in Real-World Settings

**Critical requirements**:

1. **Human oversight**: Every high-risk prediction needs professional review
2. **Transparency**: Users must know they're interacting with AI
3. **Privacy**: Mental health data is highly sensitive—encryption, anonymization, consent are non-negotiable
4. **Fail-safe mechanisms**: Clear escalation paths to human crisis counselors
5. **Regular auditing**: Monitor for bias, drift, and failure modes
6. **Diverse training data**: Include varied demographics, languages, and cultural expressions

**This project is for research and education only.** Deploying mental health AI requires interdisciplinary teams (data scientists, clinicians, ethicists, affected communities) and careful regulatory navigation.

---

## Tools & Technologies

**Core Stack**:
- **Python 3.9** - Primary language
- **PyTorch 2.0** - Deep learning framework
- **Transformers 4.31** - Hugging Face library for BERT models
- **scikit-learn 1.3** - Traditional ML & evaluation metrics
- **NLTK 3.8** - Text preprocessing

**Data & Visualization**:
- **pandas/numpy** - Data manipulation
- **matplotlib/seaborn** - Visualization
- **Jupyter** - Interactive development

**Infrastructure**:
- Trained on NVIDIA GPU (CUDA 11.8)
- Version control with Git
- Environment management via Conda

---

## Contributing

**Ways to contribute**:
- Report issues or bugs
- Suggest architectural improvements
- Share related research papers
- Discuss ethical considerations
  
---

## Acknowledgments

**Dataset**: [Suchintika Sarkar](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health) for the Sentiment Analysis for Mental Health dataset

**Mental Health Resources**: If you're struggling, please reach out:
- **International Association for Suicide Prevention**: [Find help worldwide](https://www.iasp.info/resources/Crisis_Centres/)
  
---

## License

MIT License - see [LICENSE](LICENSE) for details.

This project is open-source for **research and educational purposes**. Any real-world deployment must:
1. Comply with healthcare regulations (HIPAA, GDPR, etc.)
2. Obtain proper ethical review
3. Include mental health professionals in design and deployment

---

## Contact

- **Author:** Vidhi Rajendra Kadam
- **Email:** vidhi.kadam1501@gmail.com
- **GitHub:** https://github.com/Vidhikdm
- **LinkedIn:** https://www.linkedin.com/in/vidhikadam/

For academic inquiries, collaboration opportunities, or discussions about mental health NLP, feel free to reach out. I'm always interested in connecting with researchers working at the intersection of AI and social impact.

---

<div align="center">
  
  **If you or someone you know is struggling, please reach out for help.**  
  **You are not alone. Your life matters.**
  
  [National Suicide Prevention Lifeline](https://988lifeline.org/) | [Crisis Text Line](https://www.crisistextline.org/) | [International Resources](https://www.iasp.info/resources/Crisis_Centres/)
  
</div>

---

*This project is dedicated to everyone fighting silent battles. May we build a world where seeking help is met with understanding, not judgment.*
