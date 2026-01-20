# Chatbot Intent Detection with LoRA Fine-Tuning

This repository presents a **chatbot intent detection model** fine-tuned using **LoRA (Low-Rank Adaptation)** for **parameter-efficient training**.  
The goal of the project is to classify user utterances into predefined intents, a common task in conversational AI systems.

The model was fine-tuned on a pretrained transformer and published on **Hugging Face Hub** for easy inference and reuse.

## Project Overview

- **Task:** Intent detection (multi-class classification)
- **Domain:** Chatbot / Conversational AI
- **Approach:** LoRA fine-tuning (PEFT)
- **Frameworks:** Hugging Face Transformers, PEFT, PyTorch
- **Training environment:** Google Colab (GPU)
- **Deployment:** Hugging Face Hub

## Dataset

The model was fine-tuned on the **[tanaos/synthetic-intent-classifier-dataset-v1](https://huggingface.co/datasets/tanaos/synthetic-intent-classifier-dataset-v1)** dataset.  
It contains **~11,500 synthetic examples** of user utterances labeled with 12 intents, covering common chatbot interactions.

### Intent categories

| Label ID | Category           | Description |
|---------:|--------------------|-------------|
| 0 | greeting | Greeting or saying hello. |
| 1 | farewell | Saying goodbye or farewell. |
| 2 | thank_you | Expressing gratitude or thanks. |
| 3 | affirmation | Agreeing or confirming something. |
| 4 | negation | Disagreeing or denying something. |
| 5 | small_talk | Engaging in casual or light conversation with no specific purpose. |
| 6 | bot_capabilities | Inquiries about the bot's features or abilities. |
| 7 | feedback_positive | Providing positive feedback about the bot, service, or experience. |
| 8 | feedback_negative | Providing negative feedback about the bot, service, or experience. |
| 9 | clarification | Asking for clarification or more information about a previous statement or question. |
| 10 | suggestion | Offering a suggestion or recommendation for improvement. |
| 11 | language_change | Requesting a change in the language being used by the bot or information about language options. |

## Model & Training

- **Base model:** `bert-base-uncased`
- **Fine-tuning method:** LoRA (Parameter-Efficient Fine-Tuning)
- **Number of intents:** 12
- **Loss function:** Cross-Entropy Loss
- **Early stopping:** enabled
- **Best validation accuracy:** ~0.94

LoRA was used to significantly reduce the number of trainable parameters while maintaining high performance, making the training more efficient and lightweight.

## Results

| Epoch | Training Loss | Validation Loss | Accuracy |
|------:|---------------|------------------|----------|
| 1 | 0.3868 | 0.2981 | 0.913 |
| 2 | 0.3253 | 0.2404 | 0.930 |
| 3 | 0.1870 | 0.2406 | 0.930 |
| 4 | 0.1854 | 0.2339 | **0.936** |
| 5 | 0.2050 | 0.2336 | 0.936 |
| 6 | 0.1580 | 0.2451 | 0.935 |

### Confusion Matrix (Test Set)

![Confusion Matrix](assets/confusion_matrix.png)

## Hugging Face Model

The fine-tuned model is available on Hugging Face Hub:

**[Model on Hugging Face](https://huggingface.co/agataskrzyniarz/intent-detection-chatbot)**

## Inference

You can test the model using the provided script.

### Requirements

```
pip install transformers torch
```

### Example usage

```
python inference.py
```

or inside Python:

```
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="agataskrzyniarz/intent-detection-chatbot"
)

classifier("Can you explain it to me?")
```

## Training Notebook

The full fine-tuning process is documented in the Google Colab notebook, including:
- data loading and preprocessing
- tokenization
- LoRA configuration
- training & evaluation
- analysis of results

### Requirements

```
pip install -r requirements.txt
```

```bash
python inference.py
