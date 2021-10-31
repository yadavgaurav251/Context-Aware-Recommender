# Context-Aware-Recommender
[![GitHub stars](https://img.shields.io/github/stars/yadavgaurav251/Context-Aware-Recommender?style=social&label=Star&maxAge=2592000)](https://GitHub.com/yadavgaurav251/Context-Aware-Recommender/stargazers/)&nbsp;&nbsp;[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fyadavgaurav251%2FContext-Aware-Recommender&count_bg=%2379C83D&title_bg=%23555555&icon=deezer.svg&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
## ⛳️&nbsp; Motivation
> “If a person is presented with too many options, they are less likely to buy.”

On the Internet, where the number of choices is overwhelming, there is a need to filter, prioritize and efficiently deliver relevant information. Recommender systems help drive user engagement on platforms by generating personalized recommendations based on a user’s past behaviour.

The main motivation behind the project was to make a recommender system which is contextually aware since the context can massively change the result. While researching for the project we were not able to find any open sourced contextual recommender systems which only increased our fascination to implement one and the open-source it later.

Context cannot be taken lightly for recommender system, for instance, think you visited a hotel during summers which had everything great but their Air Conditioners were not working then you will leave a bad rating for the place but if you had visited the same hotel during winters the AC wouldn’t have mattered and you would have given them a great rating. 

Netflix famously sponsored a competition, offering a grand prize of $1,000,000 to return recommendations that were 10% more accurate than those offered by the company's existing recommender system. Experts predict that Amazon approximately gets 25% of its sales through recommendations made to users and it is here we believe things like context-awareness would shine.  So to leave out context-awareness can be a massive mistake.

## 🚀&nbsp; Installation

###  Installing Required Files 
```bash
git clone https://github.com/yadavgaurav251/Context-Aware-Recommender.git
cd Context-Aware-Recommender/
conda create --name recommender python=3.8
conda activate recommender
pip install -r requirements.txt
```
### Django Server
```bash
cd Context-Aware-Recommender/UI/
python manage.py runserver

```
## Screenshots

Movie Recommendation on Independence Day - 

![Recommendation on Independence Day](https://github.com/yadavgaurav251/Context-Aware-Recommender/blob/main/Docs/Assets/screenshot_batman.jpg)

## 📖&nbsp; Project Design and Diagram:


![graph](https://github.com/yadavgaurav251/Context-Aware-Recommender/blob/main/Docs/Assets/plan-2.png)


## 🛠&nbsp; Features

- [x] Content-Based Recommender
- [x] Collaborative Filtering Recommender System
- [x] Hybrid Recommender System 
- [x] Context-Aware Recommender System
- [x] Merging Hybrid And Context-Aware Recommender System
- [x] Django Based Interface
- [x] Integrating Django and Recommender System 

## 🛠&nbsp; Features To Implemented

- [ ] Chrome Extension
- [ ] Reinforcement Learning 
- [ ] Mobile App
- [ ] New Webapp Design

## 🤝&nbsp; Found a bug? Missing a specific feature?

Feel free to **file a new issue** with a respective title and description on the [yadavgaurav251/Context-Aware-Recommender](https://github.com/yadavgaurav251/Context-Aware-Recommender) repository. If you already found a solution to your problem, **we would love to review your pull request**! 

## 📘&nbsp; License
The Project is released under the under terms of the [MIT License](LICENSE).
