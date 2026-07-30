This file is to analyse the experiment setting, assumption(precise), and limitation

### Definition of TypiClust
A kind of low-budget active learning strategy which selects **diverse** and **typical** samples.
**Diversity** denotes that samples selected should cover more classes, should not be redundant samples which belongs to several dominant classes.
**Typicality** denotes that samples should stand for a small group of data.
TypiClust divides the dataset into multiple clusters and only choose samples from each cluster to achieve diversity.
In the mean time, in each cluster, it chooses the sample whose density is the highest.

### Assumption of TypiClust

