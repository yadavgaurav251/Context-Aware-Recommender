# Context-Aware-Recommender

## Motivation
> “If a person is presented with too many options, they are less likely to buy.”

On the Internet, where the number of choices is overwhelming, there is need to filter, prioritize and efficiently deliver relevant information. Recommender systems help drive user engagement on platforms by generating personalized recommendations based on a user’s past behaviour.
The main motivation behind the project was to make a recommender system which is contextually aware since the context can change the result in a massive way. While researching for the project we were not able to find any open sourced contextual recommender systems which only increased our fascination to implement one and the open source it later.

Netflix famously sponsored a competition, offering a grand prize of $1,000,000 to return recommendations that were 10% more accurate than those offered by the company's existing recommender system and it is here we believe things like context would shine. Experts predicts that Amazon approximately gets 25% of its sales through recommendations made to users.

Context cannot be taken lightly for recommender system, for instance think you visited a hotel during summers which had everything great but their Air Conditioners were not working then you will leave a bad rating for place but if you had visited the same hotel during winters the AC wouldn’t have mattered a would have gotten a great rating . 

## Installation

### Installing Required Files 
```bash
git clone https://github.com/yadavgaurav251/Context-Aware-Recommender.git
cd Context-Aware-Recommender/
conda create --name recommender python=3.8
conda activate recommender
pip install -r requirements.txt
```
### Django Serve
```bash
cd Context-Aware-Recommender/UI/
python manage.py runserver

```

## Project Design and Diagram:

![graph](https://github.com/yadavgaurav251/Context-Aware-Recommender/blob/main/Docs/Assets/plan-2.png)

## Features

- [x] Content Based Recommender
- [x] Collaborative Filtering Recommender System
- [x] Hybrid Recommender System 
- [x] Context-Aware Recommender System
- [x] Merging Hybrid And Context-Aware Recommender System
- [x] Django Based Interface
- [ ] Integrating Django and Recommender System 


