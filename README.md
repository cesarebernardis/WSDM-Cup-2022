# Recommender Systems Framework for Python 3.6

Developed by Maurizio Ferrari Dacrema and Cesare Bernardis, PhD candidates at Politecnico di Milano.

### Project structure

#### Base
Contains some most basic modules

##### Evaluation
Module used to evaluate a recommender object, computes various metrics.
* Accuracy metrics: ROC_AUC, PRECISION, RECALL, MAP, MRR, NDCG, F1, HIT_RATE, ARHR
* Beyond-accuracy metrics: NOVELTY, DIVERSITY, COVERAGE

Its main implementation is the SequentialEvaluator.
The evaluator takes as input the URM agains which you want to test the recommender, then a list of cutoff values (e.g., 5, 20) and, if necessary, an object to compute diversity.
The function evaluateRecommender will take as input only the recommender object you want to evaluate and return both a dictionary in the form {cutoff: results}, where results is {metric: value} and a well-formatted printable string

```python

    from Base.Evaluation.Evaluator import SequentialEvaluator

    evaluator = SequentialEvaluator(URM_test, [5, 20])

    results_run_dict, results_run_string = evaluator.evaluateRecommender(recommender)

    print(results_run_string)

```


##### Similarity
The similarity module allows to compute the item-item or user-user similarity.
It is used by calling the Compute_Similarity class and passing which is the desired similarity and the sparse matrix you wish to use.

It is able to compute the following similarities: Cosine, Adjusted Cosine, Jaccard, Tanimoto, Pearson and Euclidean (linear and exponential)

```python

    similarity = Compute_Similarity(self.URM_train, shrink=shrink, topK=topK, normalize=normalize, mode = similarity)

    W_sparse = similarity.compute_similarity()

```






