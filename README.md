# [WIP] Relevant tasks search

Searching for relevant tasks on
[Stepik](https://stepik.org/users/30844594/courses) and
[Hyperskill](https://hyperskill.org/onboarding/) platforms.

## Few words about project

In this project, we experimented with different approaches
to finding similar problems by their statements.

We took TF-IDF vectorization and pre-trained sentence BERT
model as a baselines.

## Our approaches

### 1. T5 generative model.

We used RuT5 generative model. We passed problem statement 
to the model and asked to generate a section and course 
title. Hypothesis was that hidden states in the model 
would represent the task quite well.

Script for fine-tuning can be found [here](src/tools/rut5_finetuning.py)

### 2. Sentence BERT approach

We used Sentence Transformers framework and took pre-trained 
model.
We passed pairs of similar tasks (from one Stepik’s lesson) 
to the model and tried to bring the embeddings closer 
together.
If tasks are from one Stepik’s lesson => their similarity 
equals 1, else 0.

Script for fine-tuning can be found [here](src/tools/sbert_finetuning.py)

## Results

First 4 columns relate to **inner** test, last 4 relate to **outer** test 

| model                     | Precision  | MAP@1  | MAP@3  | MAP@5  | Precision  | MAP@1  | MAP@3  | MAP@5  |
|---------------------------|:----------:|:------:|:------:|:------:|:----------:|:------:|:------:|:------:|
| TF-IDF                    | 0.4970     | 0.6364 | 0.6027 | 0.5656 | 0.3235     | 0.3652 | 0.3565 | 0.3464 |
| Sentence BERT pre-trained | 0.6606     | 0.7879 | 0.7508 | 0.7190 | 0.4261     | 0.5304 | 0.4932 | 0.4694 |
| RuT5 fine-tuned           | 0.5030     | 0.6364 | 0.6077 | 0.5743 | 0.2557     | 0.3304 | 0.3005 | 0.2849 |
| Sentence BERT fine-tuned  | 0.7273     | 0.8485 | 0.8182 | 0.7879 | 0.4557     | 0.5391 | 0.5072 | 0.4911 |

## Links

[Sentence transformers documentation](https://www.sbert.net/index.html)

[Huggingface transformers](https://huggingface.co/transformers/)