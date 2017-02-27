# Q/A System

### Introduction
Simple Q/A System that uses pre-calculated similarities to provide ranked responses to questions. Final rank is calculated based on the ranked votings of various similarity measures. Mean recall over dataset is 0.85 and mean reciprocal rank is 0.70.

### Installation
```git clone https://github.com/joostgp/qa_rank```

### Requirements
Spacy
Numpy
Pandas
Sklearn

### Usage
To pre calculate and evaluate ranks: ```python precalculate.py```
To use demo: ```python demo.py```  
