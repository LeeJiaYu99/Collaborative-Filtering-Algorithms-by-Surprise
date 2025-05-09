## Collaborative Filtering Algorithms with Surprise

### Recommender System
Recommender systems help users discover relevant content by analyzing patterns in past user-item interactions. They range from basic heuristic-based methods to complex machine learning and deep learning models.

In this repository, we will explore collaborative filtering (CF), which is one of the most widely used techniques in recommender systems using the Surprise library, a Python Scikit for building recommender systems.

### Collaborative Filtering
Collaborative filtering relies on the idea that users who have the same preference in previous selection will likely to share the similar behaviour in choosing the next product. Similarly, the products that have some sort of commonality will be preferred by the same user. Collaborative filtering focuses purely on user-item interaction data (e.g. ratings) without needing side information like product descriptions or user profiles (like how recommender system is built using content-based filtering).

Here, we divide collaborative filtering algorithms in Surprise library based on these three categories:

#### **Memory-Based CF**
Memory-based CF uses similarity measures (e.g. Pearson correlation, cosine similarity) to find neighbors of either similar users (user-based CF) or similar items (item-based CF). The rating for a target item is then predicted by aggregating the ratings from the neighbors, weighted by similarity.

**KNNBaseline**

KNN with a baseline predictor that accounts for user/item biases.

$$ \hat{r}_{ui} = b_{ui} + \frac{\Sigma_{v \in N^{k}_{i} (u)} sim(u, v) \cdot (r_{vi} - b_{ui})}{\Sigma_{v \in N^{k}_{i} (u)} sim(u, v)}$$

**KNNBasic**

A basic k-nearest neighbor model using user-user or item-item similarity.

$$ \hat{r}_{ui} = \frac{\Sigma_{v \in N^{k}_{i} (u)} sim(u, v) \cdot r_{vi}}{\Sigma_{v \in N^{k}_{i} (u)} sim(u, v)}$$

**KNNWithMeans**

KNN with rating means subtracted to normalize.

$$ \hat{r}_{ui} = \mu_u + \frac{\Sigma_{v \in N^{k}_{i} (u)} sim(u, v) \cdot (r_{vi} - \mu_v)}{\Sigma_{v \in N^{k}_{i} (u)} sim(u, v)}$$

**KNNWithZScore**

KNN using Z-score normalization for more standardized comparisons.

$$ \hat{r}_{ui} = \mu_u + \sigma_u \frac{\Sigma_{v \in N^{k}_{i} (u)} sim(u, v) \cdot (r_{vi} - \mu_v) / \sigma_v}{\Sigma_{v \in N^{k}_{i} (u)} sim(u, v)}$$

$\hat{r}_{ui}$: Predicted ratings (in the equation here $u$ is user-perspective and $v$ is of product-perspective)
\
$k$: The (max) number of neighbours to take into account for aggregation 
\
$sim$: Similarity measures (eg. cosine similarity, Pearson, etc.)

**SlopeOne**

Predicts based on the average difference in ratings between items across users. Computation for that particular target only involve the relevant user/item (e.g. the item that have at least one common user).

Pros:
+ Simple to implement by making direct use of actual ratings

Cons:
- Computationally expensive for large datasets
- Requires full user-item matrix, so cannot handle cold-start well and hard to scale

#### **Model-Based CF**

Model-based CF uses machine learning or matrix factorization (MF) techniques to generalize from past interactions and predict unknown ratings. The main idea is to decompose the user-item matrix into lower-dimensional latent feature representations. 

Approximation of reduced matrices:

$$ R \approx U_k\Sigma_k V_k^T$$ 

Absorbing $\Sigma$ into $U$ yields lower-dimensional latent feature representations of $(u^\prime_u)^T v_i$.

$$\hat{r}_{ui} = \Sigma_{k} u_{uk}s_{kk}v_{ki} = \Sigma_{k} (u_{uk} s_{kk}) v_{ki} = \Sigma_{k} u^\prime_{uk}v_{ki} = (u^\prime_u)^T v_i$$

Let vectors $u^\prime_u = p_u$ and $v_i = q_i$,

$$\hat{r}_{ui} = p_u^T q_i$$

Stochastic gradient descent (SGD) and alternating least squares (ALS) are optimization techniques that are typically used to learn the latent factors in matrix factorization models.

**SVD (Singular Value Decompostion)**

Matrix factorization technique that reduces the rating matrix to two low-rank matrices (user and item latent factors).

$$\hat{r}_{ui} = \mu + b_u + b_i + p_u^T q_i$$

Where:
- $\mu$: Global average rating
- $b_u$: Bias of user $u$ (how much higher/lower the user tends to rate)
- $b_i$: Bias of item $i$ (how much more popular the item is)

**SVD++**

Extension of SVD that includes implicit feedback (e.g. items the user has interacted with).

$$\hat{r}_{ui} = \mu + b_u + b_i + p_u^T (q_i + |I_i|^{-\frac{1}{2}} \Sigma_{j \in I_i} y_j)$$

Where $y_j$ stands for a new set of factors that capture implicit ratings.

**NMF (Non-negative Matrix Factorization)**

Similar to SVD but constrained to non-negative latent factors, useful when negative values don't make sense.

**CoClustering**

Clusters users and items simultaneously into co-clusters, predicts based on average rating per co-cluster and works well for block-like rating matrices.

$$ \hat{r_{ui}} =\bar{C_{ui}} + (\mu_u - \bar{C_u}) + (\mu_i - \bar{C_i})$$

where $\bar{C_{ui}}$ is the average rating of co-cluster $C_{ui}$, $\bar{C_u}$ is the average rating of $u$'s cluster and $\bar{C_i}$ is the average rating of $i$'s cluster.

Pros
+ More scalable and efficient on sparse data as they learn latent representations (embeddings), 

Cons
- Requires retraining when there is new data
- May overfit without regularization
- Cold-start issues remain

#### **Basic Predictors**
These algorithms serve as simple baselines, using statistical estimates without collaborative filtering logic. They are still useful to compare the accuracies with other more advanced algorithms.

**NormalPredictor**

Assumes ratings follow a normal distribution and samples predictions accordingly.

The prediction is generated from a normal distribution $N(\hat{\mu}, \hat{\sigma^2}$) where $\hat{\mu}$ and $\hat{\sigma^2}$ are estimated from the training data using Maximum Likelihood Estimation:

$$ \hat{\mu} = \frac{1}{|R_{train}|} \Sigma_{r_{ui} \in R_{train}} r_{ui}$$

$$ \hat{\sigma} = \sqrt{\Sigma_{r_{ui} \in R_{train}} \frac{(r_{ui} - \hat{\mu})^2}{|R_{train}|}}$$

**BaselineOnly**

Predicts ratings using global mean and user/item bias terms.

$$ \hat{r_{ui}} = b_{ui} = \mu + b_u + b_i $$

Pros:
+ Fast and useful as baselines
+ Useful when data is too sparse or noisy

Cons: 
- Ignores user-item interactions so cannot personalize recommendations

---
### Notebook Breakdown
- Dataset loading & Preprocessing
    - Dataset: Amazon beauty related product reviews (2M+ ratings)
    - Surprise Reader class to parse the ratings.
    - Applly Cross-validationfrom Surprise for evaluating different algorithms.
- Results & Observations
    - Memory-based methods (all KNN-inspired algorithms and SlopeOne) failed to complete the runs due to **memory constraints** of hardware.
    - SVD++ performed the best based on MAE while if based on RMSE, BaselineOnly performed best.

 ### Takeaway

Either using power of matrix factorization, or simply a statistical estimates, a recommender system can be constructed as recommendations are really probabilistic. 

The choice of algorithms relies on the goal to achieve, such as whether to create a system to recommend the globally popular ones, or the one that is closer to user preference. Besides, it also depends on data sparsity, distribution, size, and user behavior patterns.

**No one-size-fits-all.**