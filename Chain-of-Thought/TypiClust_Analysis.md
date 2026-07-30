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
- Although this paper claims that it can work well on human-crafted long-tailed Cifar10 dataset. However it is not verified on real world imbalanced dataset. It's not convincing.
<img width="772" height="337" alt="image" src="https://github.com/user-attachments/assets/1ed23a59-4663-4f92-b263-816e8f2c3305" />

- Although Clustering can obtain high diversity. But it can't guarantee that it will cover all classes.

### Exploring improvement point
Q1: Since we assume that class coverage is very important. What if we know the ground truth label of the complete dataset, and select the center of each ground truth class cluster, will this method rules the low-budget regime?

Then I test it on CIFAR100 dataset:
<img width="2096" height="757" alt="image" src="https://github.com/user-attachments/assets/ca688868-5558-433e-a4a2-46486c8af193" />
The answer is no. In the 'from scratch setting', typiclust_oracle is better than typiclust but worse than maxherding after 200 samples. In linear probe training, initially, it is significantly better than maxherding and orginal typiclust. however it is worse than maxherding after 400 samples. 
I try to explain this:
In linear probe setting. the input space is the feature space of DINO V3. In this space, features belong to the same class are clustered well, and easy to separate. In this case, we need to do class coverage at first step to build a coarse but approximately accurate decision boundary. then we need more data close to the decision boundary to refine the decision boundary. When we use typiclust, the oracle class coverage can help typiclust_oracle cover all classes in a fast manner, that's why it rules the first three rounds. However, after three rounds, the model needs points close to the boundary to refine it. typiclust_oracle can still only provide points close to the cluster centre. In contrast, Maxherding try to cover the whole data distribution. Therefore, it has higher probability to choose points close to the decision boundary.

In 'from scratch' setting. The sample number is not enough for initializing the model. the advantage brought by class balancd is overwhelmed. 
Also, the cifar100 is clean and class balanced dataset. we typiclust and maxherding can cover the class well. So it is necessary to test these methods on extremely class imbalance dataset.
<img width="2221" height="787" alt="image" src="https://github.com/user-attachments/assets/89e8239b-020e-4674-beaf-9044838c311f" />

<img width="1182" height="713" alt="image" src="https://github.com/user-attachments/assets/f0d4a37f-e5e3-43cc-ac01-5678dce28cef" />
<img width="1182" height="713" alt="image" src="https://github.com/user-attachments/assets/0f1e8b15-3b03-4906-988c-9a9c68ea831b" />
<img width="1782" height="1126" alt="image" src="https://github.com/user-attachments/assets/2c3c59a5-bae7-4693-8f13-51e77bff0003" />

I designed a new method called psedo_quota-xx. It uses external model like CLIP to generate pseudo label for all samples. and calculate typicality in each pseudo clusters.

The result shows that forced class balance can improve the performance of typiclust and maxherding. but not very much.

I then tested on real-world dataset HAM10000, <img width="600" height="292" alt="image" src="https://github.com/user-attachments/assets/e6fe2c6e-7689-45d3-a1f7-5f08f1ab1367" /> <img width="494" height="404" alt="image" src="https://github.com/user-attachments/assets/8468c22d-44a6-446a-ae4d-c67b6946fd9d" /> <img width="395" height="265" alt="image" src="https://github.com/user-attachments/assets/343763e0-16b8-4206-8267-383cf1919bcc" />

<img width="2221" height="787" alt="image" src="https://github.com/user-attachments/assets/fdc1d796-c443-42c5-b9d5-61b0ffa68e36" />
<img width="2221" height="787" alt="image" src="https://github.com/user-attachments/assets/7c135a37-90f7-4d53-a4a7-96ea8b352417" />

On this dataset, both maxherding and typiclust failed.  Typiclust_oracle also failed. 
In this case, class coverage is not the dominant factor 
<img width="1181" height="713" alt="image" src="https://github.com/user-attachments/assets/6ccedb0a-6647-4aea-89b4-1213b8873f69" />
<img width="1181" height="713" alt="image" src="https://github.com/user-attachments/assets/7ca5b30c-daf2-4ab0-8645-3794a89285b2" />












  

