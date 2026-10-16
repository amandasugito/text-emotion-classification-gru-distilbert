# 🧠 Text Emotion Classification Using Bidirectional GRU and DistilBERT

## 📌 Project Overview

This project explores text emotion classification using Natural Language Processing (NLP) and deep learning techniques. It compares two approaches: a Bidirectional Gated Recurrent Unit (GRU) model with an embedding layer and a pretrained DistilBERT transformer model.

The project covers exploratory data analysis (EDA), text preprocessing, model development, and validation-based evaluation across multiple emotion categories.

## 🎯 Project Objectives

- Explore the characteristics and distribution of emotion-labeled text data.
- Identify data quality issues and class imbalance.
- Preprocess text to reduce noise and prepare it for deep learning.
- Build a Bidirectional GRU model with an embedding layer.
- Fine-tune a pretrained DistilBERT-based classification model.
- Analyze training and validation performance.

## 📂 Dataset

The project uses a text emotion dataset provided in `2B.zip`, containing the file `text_emotions.tsv`.

The dataset includes text samples labeled with six emotion categories:

- Anger
- Fear
- Joy
- Love
- Sadness
- Surprise

### 🔎 Exploratory Data Analysis

The EDA examines the dataset structure, emotion class distribution, and text length distribution.

Key observations documented in the notebook include:

- **Class imbalance:** Joy and sadness have relatively large numbers of samples, while surprise has the fewest.
- **Variable text length:** Most texts are short, while a smaller number are considerably longer.
- **Potential classification challenges:** Minority classes may be harder to predict reliably, particularly when emotion categories have overlapping meanings.

## 🧹 Text Preprocessing

The preprocessing pipeline includes:

- Lowercasing text.
- Removing URLs.
- Removing numbers.
- Removing punctuation and special characters.
- Tokenization.
- English stopword removal.
- Lemmatization.
- Removing samples that become empty after preprocessing.

For the Bidirectional GRU model, text is converted into integer sequences, then padded or truncated to a fixed length of 40 tokens.

For DistilBERT, the cleaned text is tokenized using `DistilBertTokenizerFast` with a maximum sequence length of 40 tokens.

## 🧠 Model Development

### 1. 🔄 Bidirectional GRU

The first model uses a Keras Sequential architecture consisting of:

- An embedding layer with an embedding dimension of 64.
- A bidirectional GRU layer with 1,024 units and sequence outputs.
- A second bidirectional GRU layer with 512 units.
- A dropout layer with a rate of 0.5.
- A softmax output layer for multiclass emotion classification.

The model is compiled using the Adam optimizer and categorical cross-entropy loss.

### 2. 🤗 DistilBERT

The second approach uses `distilbert-base-uncased`, a pretrained transformer model for text representation and classification.

The notebook configures the model with:

- A sequence classification head for six emotion categories.
- A maximum sequence length of 40 tokens.
- Frozen pretrained DistilBERT parameters.
- A training configuration specifying 15 epochs.
- Training and validation datasets created using an 80:20 split before the notebook applies sample limits.

The notebook also includes training and validation loss analysis.

## 📊 Validation Results

The following accuracy values are reported in the notebooks:

| Model | Reported Validation Accuracy |
|---|---:|
| Bidirectional GRU | 92.26% |
| DistilBERT | 63.75% |

These values are recorded validation results, not results from an independent test set.

The notebooks also include confusion matrix and classification report analyses to examine prediction performance across individual emotion categories.

### 📈 Interpretation

The notebook analysis indicates that the Bidirectional GRU achieved higher reported validation accuracy in this experiment. However, the results should be interpreted cautiously because the models use different implementations and sample configurations.

The analysis also identifies lower recall for some minority emotion categories, including love and surprise, suggesting that overall accuracy alone does not fully represent classification performance.

## 🛠️ Tools and Technologies

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- NLTK
- Scikit-learn
- TensorFlow / Keras
- PyTorch
- Hugging Face Transformers

## 📁 Project Structure

- `README.md` — Project overview, methodology, and reported results.
- `emotion-classification-bidirectional-gru.ipynb` — EDA, text preprocessing, Bidirectional GRU development, and validation analysis.
- `emotion-classification-distilbert.ipynb` — DistilBERT implementation, training configuration, and validation analysis.
- `data/2B.zip` — Original dataset archive.

## 🚀 How to Run

1. Clone or download this repository.
2. Open the notebooks in Google Colab or a compatible Jupyter environment.
3. Upload `2B.zip` to the environment if it is not already available.
4. Run the notebook cells in sequence.
5. Review the training history, validation loss, confusion matrix, and classification report.

The notebooks require the relevant Python libraries and access to the pretrained DistilBERT model.

## 👩‍💻 Author
**Amanda Sugito**

Deep Learning academic project at BINUS University.

## 🔗 Connect with Me
- LinkedIn: Amanda Sugito | www.linkedin.com/in/amanda-sugito
- GitHub: amandasugito | https://github.com/amandasugito

## ⚠️ Limitations

- The dataset has an imbalanced distribution across emotion classes.
- The reported metrics are based on validation data rather than an independent test set.
- DistilBERT uses a limited subset of the available training and validation samples in the notebook.
- Further evaluation with a consistent data split, reproducible training procedure, and independent test set is needed for a fair comparison.

This project is an academic deep learning exercise and should not be interpreted as a production-ready emotion detection system.
