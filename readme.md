# Q/A System

### Introduction
Simple Q/A System that uses pre-calculated similarities to provide ranked responses to questions. Final rank is calculated based on the ranked votings of various similarity measures. Mean recall over dataset is 0.85 and mean reciprocal rank is 0.70.

### Installation
```git clone https://github.com/joostgp/qa_rank```

### Requirements
- Spacy
- Numpy
- Pandas
- Sklearn

### Usage
- To pre calculate and evaluate ranks (takes 5min): ```python precalculate.py```
- To use demo: ```python demo.py```

### Example output:
```
Loading and preprocessing data from Acme_dataset.xlsx
Calculating rank using keyword matching
Done! Scoring: mean rank = 0.65, mean recall = 0.78
Calculating rank using count vector similarity
Done! Scoring: mean rank = 0.66, mean recall = 0.81
Calculating rank using tf-ifd vector similarity
Done! Scoring: mean rank = 0.70, mean recall = 0.83
Calculating rank using tf-ifd vector similarity on first sentence
Done! Scoring: mean rank = 0.62, mean recall = 0.74
Calculating rank using noun chunk matching
Done! Scoring: mean rank = 0.51, mean recall = 0.62
Calculating rank using mean word vector similarity
Done! Scoring: mean rank = 0.61, mean recall = 0.74
Calculating rank ensemble
Done! Scoring: mean rank = 0.71, mean recall = 0.84
Storing results in Acme_dataset_pre.txt
```  
