# Lab 8–9: Ensemble Methods

## 1. Implementation of AdaBoost

### (a)
Implement your own version of AdaBoost algorithm using trees as a base classifier – use available implementations of decision trees.

#### Training
1. Define initial weights: \(w_i = 1/n\), \(i = 1, \dots, n\).
2. For \(k = 1, \dots, B\):
   - Build classifier \(f_k\) with weights \(w_i\).
   - Compute weighted classification error:
     \[
     \varepsilon_k = \sum_{i=1}^{n} I[f_k(x_i) \ne y_i] w_i.
     \]
   - If \(\varepsilon_k \notin (0, 0.5)\), handle this case appropriately and state clearly what your implementation should do.
   - Compute scaling factor:
     \[
     \beta_k = \frac{\varepsilon_k}{1-\varepsilon_k}.
     \]
   - Add pair \((f_k,\beta_k)\) to the ensemble.
   - For \(i=1,\dots,n\):
     - If \(f_k(x_i)=y_i\), set \(w_i=w_i\beta_k\).
   - Normalize weights:
     \[
     w_i=\frac{w_i}{\sum_{j=1}^{n} w_j}.
     \]

#### Prediction
For a new observation \(x\):
\[
\hat y(x)=\arg\max_y \sum_{k=1}^{B} I[f_k(x)=y]\log\left(\frac1{\beta_k}\right).
\]

### (iii)
Define clearly what should happen when:
- \(\varepsilon_k = 0\)
- \(\varepsilon_k \ge 0.5\)

Explain why these cases are problematic for AdaBoost.

### (iv)
Compare AdaBoost with:
- base trees of depth 1,
- deeper trees.

Comment on:
- influence of base learner depth on results,
- training time.

### (v) Theoretical Questions
A. Explain why a classifier with smaller \(\varepsilon_k\) should receive a larger weight in the final prediction.

B. Assume that 999 classifiers have error \(\varepsilon_k = 0.4\) and one classifier has error \(\varepsilon_{1000}=10^{-8}\). Compute the corresponding voting weights and comment on whether one almost perfect classifier dominates the final decision.

C. Why are very simple base classifiers, such as decision stumps, often sufficient in AdaBoost?

---

## 2. Comparison of Ensemble Methods

### (a)
Compare:
- Single tree
- Bagging
- Boosting (AdaBoost, your implementation)
- Gradient Boosting
- XGBoost
- Random Forest

### (b)
Consider:
- one real classification dataset,
- one artificial dataset.

Artificial dataset:

- Generate \(X_1,\dots,X_{10}\sim N(0,1)\).
- Let \(\chi^2_{10}(0.5)\) denote the median of the chi-squared distribution with 10 degrees of freedom.
- Set:
  \[
  Y=1 \quad \text{if} \quad \sum_{j=1}^{10}X_j^2>\chi^2_{10}(0.5),
  \]
  otherwise \(Y=-1\).
- Generate:
  - training data size 2000,
  - testing data size 10000.

### (c)
Why do we expect \(Y\) to be balanced in this artificial dataset?

### (d)
Describe the shape of the Bayes decision boundary. Which methods should be better suited to approximate it, and why?

### (e)
Make plots showing how training and testing error change with the number of iterations/trees for:
- Boosting
- Bagging
- Random Forest
- Gradient Boosting
- XGBoost

### (f)
Compare methods using the same train/test split.

### (g)
For each method, comment whether improvement over a single tree comes mainly from:
- variance reduction,
- bias reduction,
- or both.

### (h)
Explain why Random Forest may outperform Bagging, although both average many trees.

---

# Lab 10: Feature Selection

## 1. Data Generation

Consider parameters:
- \(n\): training size,
- \(p\): total number of features,
- \(k\): number of significant features.

### (a) Dataset 1

- Generate independent features:
  \[
  X_1,\dots,X_p \sim N(0,1).
  \]
- Let \(\chi^2_k(0.5)\) denote the median of the chi-squared distribution with \(k\) degrees of freedom.
- Set:
  \[
  Y=1 \text{ if } \sum_{j=1}^{k}X_j^2 > \chi^2_k(0.5),
  \]
  otherwise \(Y=0\).
- Generate training data of size \(n\).

### (b) Dataset 2

- Generate independent features:
  \[
  X_1,\dots,X_p \sim N(0,1).
  \]
- Set:
  \[
  Y=1 \text{ if } \sum_{j=1}^{k}|X_j| > k,
  \]
  otherwise \(Y=0\).
- Generate training data of size \(n\).

### Question
Are relevant variables \(X_1,\dots,X_k\) expected to be detectable by simple marginal correlation with \(Y\)? Why or why not?

---

## 2. Comparison of Feature Selection / Feature Ranking Methods

Consider:
- Random Forest variable importance:
  - mean decrease in impurity,
  - permutation importance.
- Boruta algorithm.

### (a)
Check whether the algorithms assign highest importance scores to significant variables.

### (b)
Try different values of:
- \(n\),
- \(p\),
- \(k\).

Suggested starting values:
- \(n=500\),
- \(p=50\),
- \(k=10\).

### (c)
Repeat data generation \(L=50\) times (or fewer if computationally expensive). Estimate probability of successful feature recovery.

### (d)
Fix:
- \(n=200\),
- \(p=500\),
- \(k=20\).

Let \(t\) be the number of top-ranked features.

Generate an independent test set.

Train a classifier (e.g. Random Forest) using the top \(t\) features and analyze test accuracy for:

\(
t \in \{5,10,15,20,50,100,200,300,400,500\}
\)

Generate a plot.

### (e)
Explain the role of shadow variables in Boruta. Why is comparison with the best shadow variable more meaningful than checking whether an importance score is positive?

---

# Lab 11: Support Vector Machines

## 1. Linear SVM and Logistic Regression

Generate a two-dimensional linearly separable dataset using the code provided in the lab instructions.

### Before fitting
Predict:
- Will SVM and logistic regression decision boundaries be similar or different?
- Which observations are expected to influence each method most strongly?

### Tasks

Split data into training and testing sets.

Standardize predictors before fitting.

Fit:
- Logistic Regression
- Linear SVM

For both methods:

#### (a)
Compute:
- training accuracy,
- testing accuracy.

#### (b)
Draw decision regions.

#### (c)
Compare decision boundaries.

For Linear SVM additionally draw:
- separating hyperplane,
- margin lines,
- support vectors.

### Questions

(a) Both methods use linear decision boundaries. Why are the boundaries different?

(b) Which observations become support vectors?

(c) Are support vectors necessarily misclassified observations?

(d) Does linear SVM estimate posterior probabilities \(P(Y=1\mid X=x)\)?

(e) Why does the SVM boundary depend mainly on observations close to the separating boundary, while logistic regression is influenced by all observations?

(f) How does the SVM boundary depend on parameter:
\(
C \in \{0.01,0.1,1,10,100\}
\)

(g) For each value of \(C\), report the number of support vectors.

---

## 2. Kernel SVM

Generate a nonlinear binary classification dataset:

```python
from sklearn.datasets import make_moons

X, y = make_moons(
    n_samples=300,
    noise=0.25,
    random_state=123
)
```

Split into training and testing sets.

Standardize predictors before fitting.

Fit:

- Logistic Regression
- Logistic Regression + polynomial features (degree 2)
- Logistic Regression + polynomial features (degree 3)
- Linear SVM
- SVM with polynomial kernel:
  - \(C=10\)
  - \(coef0=10\)
  - \(d \in \{2,3,4\}\)
- SVM with RBF kernel

For each method:

### (a)
Compute:
- training accuracy,
- testing accuracy.

### (b)
Draw decision regions.

### (c)
Compare decision boundary shape.

### RBF Kernel Study

Fix:
\(
C=1
\)

Fit models for:

\(
\gamma \in \{0.01,0.1,1,10,100\}
\)

For each value:

(a) Compute training and testing accuracy.

(b) Draw decision regions.

(c) Compare complexity of decision boundaries.

### Questions

(a) Why do ordinary logistic regression and linear SVM perform poorly on this dataset?

(b) Does logistic regression with polynomial features of degree 3 produce a proper decision boundary for this task?

(c) What does the RBF kernel allow SVM to do?

(d) What happens when gamma is very small?

(e) What happens when gamma is very large?

(f) Compare logistic regression with polynomial features and SVM with RBF kernel. In what sense are these approaches similar, and in what sense are they different?
