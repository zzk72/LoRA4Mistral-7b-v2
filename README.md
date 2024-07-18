---
library_name: peft
base_model: /home/zbdc/LLMs/mistral-7b-v2
---

# Model Card for LoRA-finetuned Mistral-7B

This model is a LoRA-finetuned version of Mistral-7B, aimed at recovering prompts for large language models. Only the LoRA adapter parameters are being open-sourced.

## Model Details

### Model Description

This model has been fine-tuned using LoRA (Low-Rank Adaptation) on the Mistral-7B base model to enhance its ability to generate and recover prompts for large language models. The fine-tuning process improves the model's performance in specific text generation and prompt recovery tasks.

- **Developed by:** Zhang Zhenkui
- **Model type:** Fine-tuned LLM
- **Language(s) (NLP):** English
- **License:** Apache License
- **Finetuned from model:** Mistral-7B-v2

### Model Sources

- **Repository:** https://github.com/zzk72/LoRA4Mistral-7b-v2
- **Paper [optional]:** N/A
- **Demo [optional]:** N/A

## Uses

### Direct Use

This model can be used directly for text generation and prompt recovery tasks in large language models.

### Downstream Use

Fine-tuned for specific tasks such as prompt recovery, text generation, and other NLP tasks requiring contextual understanding and text coherence.

### Out-of-Scope Use

The model is not suitable for tasks outside the domain of text generation and prompt recovery, and may not perform well on unrelated NLP tasks.

## Bias, Risks, and Limitations

The model inherits biases from the training data and the base model. It is important to consider these biases when deploying the model in sensitive applications.

### Recommendations

Users should be aware of the biases and limitations of the model. It is recommended to evaluate the model's performance in your specific use case and make adjustments as necessary.

## Training Details

### Training Data

The training data consists of a mixture of human-written and LLM-generated texts, processed to enhance the model's ability to recover and generate prompts.

### Training Procedure

#### Preprocessing

Text data was preprocessed using custom BPE tokenization to handle typographical errors effectively.

#### Training Hyperparameters

- **Training regime:** fp16 mixed precision

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The evaluation was conducted on a separate dataset comprising both human-written and LLM-generated texts.

#### Factors

Evaluation factors include text coherence, prompt recovery accuracy, and generation quality.

#### Metrics

- Accuracy
- F1 Score
- BLEU Score

### Results

The model achieved high accuracy in prompt recovery and demonstrated strong performance in text generation tasks.

#### Summary

The LoRA-finetuned Mistral-7B model is effective in generating and recovering prompts for large language models, providing enhanced performance over the base model.

## Environmental Impact

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** GPUs 2*RTX3090
- **Hours used:** 5h 

## Technical Specifications

### Model Architecture and Objective

LoRA-finetuned Mistral-7B for prompt recovery tasks.

### Compute Infrastructure

#### Hardware

GPUs used for training.

#### Software

PEFT 0.10.0, Transformers library

## Citation

**BibTeX:**

```
@misc{zhang_zhenkui_2024,
  author = {Zhang Zhenkui},
  title = {LoRA-finetuned Mistral-7B for Prompt Recovery},
  year = {2024},
  url = {https://your-repo-link}
}
```

**APA:**

Zhang Zhenkui. (2024). LoRA-finetuned Mistral-7B for Prompt Recovery. Retrieved from https://your-repo-link

## Model Card Authors

Zhang Zhenkui


### Framework versions

- PEFT 0.10.0
