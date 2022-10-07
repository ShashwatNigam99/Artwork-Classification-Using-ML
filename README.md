# ArtyLyze

<embed width="800" height="500" src="https://www.youtube.com/embed/J0ZKjrVC2QY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen>

## Introduction
- Since the advent of digitalization, millions of artworks have been digitized, opening up the world of art to countless new people. However, navigating this space for a layman is difficult due to the lack of metadata and contextual information needed to describe and understand the artwork. 
- Unless one knows title and artist of an artwork, finding the artwork is almost impossible. In this research project, we use both supervised and unsupervised models to generate metadata given an image of an artwork. 
- Specifically, given an artwork, we will try to predict the genre, time period, geography and the artist. 
- We have collected dataset of ~30k images from [WikiArt](wikiart.org) along with their above mentioned metadata.

## Problem Definition
1. *Throughout history it has been observed that artistic collaborations fuel creativity and give rise to art movements.* Our study aims to find correlations between different artistic styles spanning geographies and periods, which would help track the journey of art and how it evolved. Soft clustering approaches can help deduce influence of different factors (genre, time period, geography) on a particular work of art. Studying similarity and influence across time and geography between different art styles is a relevant research area.
2. We wish to build a model that can classify an artwork according to its genre, time period, geography etc. If possible we would like to study the features the model learns that helps it differentiate between different kinds of artworks.


## Related Work 
| Research                                                                                                                                         | Datasets                                                                                        | Models              | Predictions        | Metrics                                                     | Year |
|----------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|-----------------------|----------------------|---------------------------------------------------------------|---------|
| [Large-scale Classification of Fine-Art Paintings: Learning The Right Metric on The Right Feature](https://arxiv.org/pdf/1505.00855.pdf)             | [WikiArt](https://www.wikiart.org/)                                                                 | Classemes, Picodes, CNN | Style, Genre, Artist   | Accuracy: 63%                                                   | 2015      |
| [Compare the performance of the models in art classification](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0248414)             | Painting-91, Wikiart, Multitask Painting100K                                                         | CNN                     | Style, Genre, Artist  | <ul> <li>Painting-91: 80% <li>Wikiart: 91% <li>Multitask Painting100K: 65% | 2021      |
| [Classifying digitized art type and time period](https://www.jevinwest.org/papers/Yang2018KDDart.pdf)                                                | 300k images from: Metropolitan Museum of Art, WikiArt and Artsy                                     | CNN                     | Type, Time period     | Accuracy for type: 87%, for time: 57%                           | 2018      |
| [The Effect of Derived Features on Art Genre Classification with Machine Learning](http://www.saujs.sakarya.edu.tr/en/download/article-file/1668894) | [50 Most Influential Artists ](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time) | Random Forest           | Genre            | F1 score: 82%                                                   | 2021      |

## Dataset
Dataset of ~30k images from [WikiArt](wikiart.org) along with metadata like:
- Genre
- Artist
- Year
- Location
  
<details>
<summary>Click here to see a sample datapoint!</summary>
<pre>
{
        "title": "Picador",
        "contentId": 224238,
        "artistContentId": 223667,
        "artistName": "Picasso Pablo",
        "completitionYear": 1900,
        "yearAsString": "1900",
        "width": 810,
        "image": "https://uploads6.wikiart.org/images/pablo-picasso/picador-1900.jpg!Large.jpg",
        "height": 1280,
        "artistUrl": "pablo-picasso",
        "url": "picador-1900",
        "dictionaries": [
            417,
            502
        ],
        "location": null,
        "period": "Early Years",
        "serie": null,
        "genre": "genre painting",
        "material": null,
        "style": "Expressionism",
        "technique": null,
        "sizeX": 13.5,
        "sizeY": 21.0,
        "diameter": null,
        "auction": null,
        "yearOfTrade": null,
        "lastPrice": null,
        "galleryName": "Museum of Montserrat, Barcelona, Spain",
        "tags": "animals, horses, horsemen, hunting-and-racing, bulls, Fiction",
        "description": null
    }
</pre>
</details>
  
## Methods
We employ both supervised and unsupervised methods for this project. 
![Screenshot from 2022-10-07 13-09-51](https://user-images.githubusercontent.com/30972206/194612955-212975aa-be7a-45f0-a484-7b8f802b42c9.png)
![Screenshot from 2022-10-07 13-11-01](https://user-images.githubusercontent.com/30972206/194613037-52806952-4c98-4d47-989f-5bd84f4f023b.png)
### Supervised classification:
- Convolutional Neural Network
### Unsupervised classification: 
- K-means 
- Gaussian Mixture Models
- DBSCAN

## Results and Discussion
For the supervised task of classification use the following metrics to compare different models:

1. precision
2. recall
3. f1-score
4. top-k accuracy
5. neg_log_loss
6. roc_auc_ovr
7. Computational efficiency 
8. Time taken to reach convergence 

If we encounter imbalance in the dataset we will include the following metrics as well for model evaluation: 

1. balanced_accuracy
2. f1_weighted 

We are using the following internal measures to compare our clustering approaches since we do not have access to ground truth cluster labels: 

1. Silhouette Coefficient
2. Davies-Bouldin Index
3. Beta-CV Measure 

In our results we hope that our supervised model can correctly identify the class of each painting. We expect that the unsupervised clustering approach will shed interesting insights on the correlation between the style and work of different artists spanning different periods and geographies.

## Team
Anisha Pal, Avinash Prabhu, Meher Shashwat Nigam, Mukul Khanna, Shivika Singh
### Work Distribution
[Link to Gantt Chart](https://docs.google.com/spreadsheets/d/1LRcJbVBx7WDqoV-0qWDu1E3mkWw_PsMho9b2ql4kQBE/edit?usp=sharing)

## References
```
@misc{saleh2015largescale,
      title={Large-scale Classification of Fine-Art Paintings: Learning The Right Metric on The Right Feature}, 
      author={Babak Saleh and Ahmed Elgammal},
      year={2015},
      eprint={1505.00855},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
@article{10.1371/journal.pone.0248414,
    doi = {10.1371/journal.pone.0248414},
    author = {Zhao, Wentao AND Zhou, Dalin AND Qiu, Xinguo AND Jiang, Wei},
    journal = {PLOS ONE},
    publisher = {Public Library of Science},
    title = {Compare the performance of the models in art classification},
    year = {2021},
    month = {03},
    volume = {16},
    url = {https://doi.org/10.1371/journal.pone.0248414},
    pages = {1-16},
}
@article{ saufenbilder904964, 
    journal = {Sakarya University Journal of Science},
    year = {2021}, volume = {25}, number = {6}, pages = {1275 - 1286}, 
    doi = {10.16984/saufenbilder.904964}, 
    title = {The Effect of Derived Features on Art Genre Classification with Machine Learning}, 
    author = {Abidin, Didem} 
}
@inproceedings{Yang2018ClassifyingDA,
  title={Classifying digitized art type and time period},
  author={Sean T. Yang and Bum Mook Oh},
  year={2018}
}
```
