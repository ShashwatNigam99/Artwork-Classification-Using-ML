# ArtyLyze
![cover](https://user-images.githubusercontent.com/30972206/194464894-9d62315c-a117-4bab-8a67-a149fac6448f.jpg)

## Introduction
- Since the advent of digitalization, millions of artworks have been digitized, opening up the world of art to countless new people. However, navigating this space for a layman is difficult due to the lack of metadata and contextual information needed to describe and understand the artwork. 
- Unless one knows title and artist of an artwork, finding the artwork is almost impossible. In this research project, we use both supervised and unsupervised models to generate metadata given an image of an artwork. 
- Specifically, we will try to predict the artist name, genre, style, completion year, location of the artwork, and the time period. 
- We have collected dataset of ~30k images from wikiart.org along with their above mentioned metadata.

## Problem Definition
1. *Throughout history it has been observed that artistic collaborations fuel creativity and give rise to art movements.* Our study aims to find correlations between different artistic styles spanning geographies and periods, which would help track the journey of art and how it evolved. Soft clustering approaches can help deduce influence of different factors (genre, time period, geography) on a particular work of art. Studying similarity and influence across time and geography between different art styles is a relevant research area.
2. We wish to build a model that can classify an artwork according to its genre, time period, geography etc. If possible we would like to study the features the model learns that helps it differentiate between different kinds of artworks.


## Related Work 
| Research                                                                                                                                         | Datasets                                                                                        | Models              | Predictions        | Metrics                                                     | Year |
|----------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|-----------------------|----------------------|---------------------------------------------------------------|---------|
| [Large-scale Classification of Fine-Art Paintings: Learning The Right Metric on The Right Feature](https://arxiv.org/pdf/1505.00855.pdf)             | [WikiArt](https://www.wikiart.org/)                                                                 | Classemes, Picodes, CNN | Style, Genre, Artist   | Accuracy: 63%                                                   | 2015      |
| [Compare the performance of the models in art classification](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0248414)             | Painting-91, Wikiart, MultitaskPainting100K                                                         | CNN                     | Style, Genre, Artist  | <ul> <li>Painting-91: 80% <li>Wikiart: 91% <li>MultitaskPainting100K: 65% | 2021      |
| [Classifying digitized art type and time period](https://www.jevinwest.org/papers/Yang2018KDDart.pdf)                                                | 300k images from: Metropolitan Museum of Art, WikiArt and Artsy                                     | CNN                     | Type, Time period     | Accuracy for type: 87%, for time: 57%                           | 2018      |
| [The Effect of Derived Features on Art Genre Classification with Machine Learning](http://www.saujs.sakarya.edu.tr/en/download/article-file/1668894) | [50 Most Influential Artists ](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time) | Random Forest           | Genre            | F1 score: 82%                                                   | 2021      |

## Methods
We employ both supervised and unsupervised methods for this project. 
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
