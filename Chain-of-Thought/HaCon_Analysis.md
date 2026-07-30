This file is to analyse the experiment setting, assumption(precise), and limitation

### Definition of HaCon
A kind of low-budget one-shot active learning strategy which selects **diverse** and **typical** samples.

**Diversity** denotes that samples selected should cover more classes, should not be redundant samples which belongs to several dominant classes.

**Typicality** is defined in a different way compared to typiclust here. It defines that typical data should be **hard-to-contrast** samples in the instance discrimitive task.

### Assumption of TypiClust
- Dataset
  - The dataset doesn't have noisy labels.
  - All samples belong to the same data distribution
- Each comparison learning cluster stands for a category.
- Mechanism
  - Hard-to-contrast is high typicality.

### Limitations
- It works worse than random sampling when budget is not extremely low. "When is extremely low or when to switch to another strategy is not defined well".
- Although this paper claims that it can work well on human-crafted long-tailed Cifar10 dataset. However it is not verified on real world imbalanced dataset.
- [x] look for another limitation
