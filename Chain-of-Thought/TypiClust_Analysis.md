This file is to analyse the experiment setting, assumption(precise), and limitation

### Definition of TypiClust
A kind of low-budget active learning strategy which selects **diverse** and **typical** samples.

**Diversity** denotes that samples selected should cover more classes, should not be redundant samples which belongs to several dominant classes.

**Typicality** denotes that samples should stand for a small group of data.
TypiClust divides the dataset into multiple clusters and only choose samples from each cluster to achieve diversity.
In the mean time, in each cluster, it chooses the sample whose density is the highest.

### Assumption of TypiClust
- Dataset
  - The dataset doesn't have noisy labels.
  - All samples belong to the same data distribution
- Embedding
  - The embedding reflects semantic relations well.
- Mechanism
  - High density sample $\propto$ high error reduction 

### Limitations
- It works worse than random sampling when budget is not extremely low. "When is extremely low or when to switch to another strategy is not defined well".
- Although this paper claims that it can work well on human-crafted long-tailed Cifar10 dataset. However it is not verified on real world imbalanced dataset.
- [] look for another limitation
