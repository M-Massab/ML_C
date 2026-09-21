## **1\. The Model (The Math Formula)**

A model is fundamentally an algorithm or a mathematical formula.

* **Linear Regression** uses the equation of a straight line: $y \= mx \+ c$ (or $y \= wx \+ b$).  
* When we say we are building a "linear model," it means we are using this specific equation to predict outcomes.  
* The goal of training is to find the perfect value for the weight ($w$) so the equation outputs highly accurate predictions.

## **2\. The Cost Function (Measuring Error)**

The Cost Function measures the overall difference between the model's predictions and the actual true values across the entire dataset. The specific formula changes depending on the model you use.

**Mean Squared Error (MSE)**

Linear Regression uses MSE as its cost function:

J \= (1/2m)\* submission (y \- y\_pred)^2

* If you calculate the cost for many different values of $w$ and plot them on a graph, MSE always creates a perfect U-shaped curve or bowl (a **Convex Function**).  
* Because it is a perfect bowl, it has a single **Global Minima** (one absolute lowest point) and zero **Local Minima** (false bottoms). This guarantees the math will always converge on the single best parameter.

## **3\. Gradient Descent (The Optimizer)**

Gradient Descent is an optimization algorithm used by parametric models (like Linear Regression, Logistic Regression, and Support Vector Machines) to minimize the cost function.

It reduces the error by continuously updating the weights using this formula:

w \= w \- alpha ((partial derivative J)\\(partial w))

*(Where $\\alpha$ is the Learning Rate, and $\\frac{\\partial J}{\\partial w}$ is the partial derivative).*

**The Derivative (The Slope)**

The mathematical derivative represents the slope of the curve at any given point. For Linear Regression, it is calculated as:

partial J/ partial w \= (1/m)\* submission (y\_pred \- y)x

## **4\. Variants of Gradient Descent**

How the algorithm groups the data before taking a step down the slope:

* **Batch Gradient Descent:** Calculates the slope for the *entire* dataset, averages it, and then updates the parameters once per epoch.  
* **Stochastic Gradient Descent (SGD):** Calculates the slope for a *single, random example* and updates the weights immediately. Very fast, but bounces around wildly.  
* **Mini-Batch Gradient Descent:** The industry standard. Calculates the slope for a small chunk of data (e.g., 32 or 64 examples), averages them, and updates the weights.

## **5\. Multiple Linear Regression (Scaling Up)**

When dealing with multiple features (Square Footage, Age, Rooms), the mechanics change slightly to handle the scale:

* **Vectorization:** Instead of writing slow for loops, it uses Linear Algebra matrices and **Dot Products** to calculate all features instantly.  
* **Partial Derivatives:** The algorithm calculates the slope for one specific parameter at a time by mathematically "freezing" all the other parameters in place.  
* **Simultaneous Updates:** All weights ($w\_1, w\_2, w\_3$) calculate their slopes based on the exact same shared error, and then *all weights are updated simultaneously* before the new cost is calculated.

&nbsp;

&nbsp;

&nbsp;

**1\.  Problem with Linear Regression:**&nbsp;

&nbsp;1: Unbounded output \-\> output negative infinity to positive infinity&nbsp;

&nbsp;2: Sensitivity to outliers destroys your threshold: Example: If most of malignant tumor size is 0.8-10, and model set /fitted the threshold at 0.5 predicting size greater than 0.5 is malignant,  but a single sample with outlier value like 50, will change the models fitting and can shift model threshold to 0.9 etc.&nbsp;

![][image1]![][image2]

&nbsp;

&nbsp;

## **2\. Why Logistic Regression Uses Log Loss Instead of MSE**

**1\. The Non-Convexity Problem (False Bottoms)**

* **MSE:** Squaring the curved Sigmoid function creates a wavy, non-convex error landscape. Gradient Descent gets permanently stuck in "false bottoms" (local minima).  
* **Log Loss:** Uses logarithms to flatten the waves, restoring a perfectly smooth, single-bottomed bowl (convex) that guarantees finding the best possible weights.

**2\. Bounded Errors (Vanishing Gradients)**

* **MSE:** Because probabilities strictly live between 0 and 1, the maximum possible error is 1.0. Squaring this creates a tiny penalty, causing the model to barely update its weights and stop learning.  
* **Log Loss:** Solves this by applying a massive, exploding penalty to *confident mistakes* (e.g., predicting 0.001 when the truth is 1). This massive gradient forces the weights to correct themselves rapidly.

&nbsp;

&nbsp;

**3\. Weight/parameters impact:** A change in weight means how it effects the the odd ratio of classes. For a single feature:

&nbsp;&nbsp;z \= w · x \+ b

&nbsp;

&nbsp;&nbsp;→ A 1-unit increase in x changes the log-odds by exactly w

&nbsp;&nbsp;→ Exponentiating gives the Odds Ratio: e^w

&nbsp;

&nbsp;

**Weight value**		**e^w (Odds Ratio)**		**Real-world meaning**

`w = 0.0`			1.0			Feature has zero influence

`w =+0.5`			1.65			Each unit increase → odds of class 1                               							increase 65%

`w = −0.8`			0.45			Each unit increase → odds of class 1   								decrease 55%

`w = +2.0`			7.39			Very strong positive predictor&nbsp;&nbsp;

&nbsp;

&nbsp;

&nbsp;

**4\. Advanced Tuning & Interpretation**:&nbsp;

**1\)   Regularization:** Why Regularization Exists

* Without regularization, a model will often overfit — it memorizes the training data's noise rather than learning the true signal. Coefficients grow arbitrarily large to fit every quirk of the training set, then fail badly on new data.  
  &nbsp;  
* Regularization adds a penalty term to the loss function that discourages large weights:  
  ── Standard Loss :  L \= Binary Cross-Entropy  
  &nbsp;  
  ── L2 Regularized Loss (Ridge):  L \= Binary Cross-Entropy \+ λ · Σwᵢ²  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↑  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;penalizes large weights  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;by squaring them  
  &nbsp;  
  ── L1 Regularized Loss (Lasso): L \= Binary Cross-Entropy \+ λ · Σ|wᵢ|  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↑  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;penalizes large weights  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;by their absolute value  
  &nbsp;

  #### 

  #### 

* #### **L1 vs L2 — The Core Difference**

| Property | L1 (Lasso) | L2 (Ridge) |
| ----- | ----- | ----- |
| Effect on weights | Drives some to **exactly zero** | Shrinks all weights, none to zero |
| Built-in feature selection | ✅ Yes — sparse model | ❌ No — all features kept |
| When to use | High-dimensional data, suspect many irrelevant features | Most features matter, want stability |
| sklearn param | `penalty='l1'`, `solver='liblinear'` | `penalty='l2'` (default)&nbsp; |

**2\) The `C` Hyperparameter — sklearn's Regularization Knob**: sklearn uses `C = 1/λ` — the inverse of regularization strength.&nbsp;

—---C very small (e.g. 0.001)  →  λ very large  →  heavy regularization

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→  weights pushed hard toward 0

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→  simpler model, may underfit

&nbsp;

—---C \= 1.0 (default)          →  moderate regularization

&nbsp;

—---C very large (e.g. 1000\)   →  λ ≈ 0         →  almost no regularization

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→  weights free to grow large

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→  complex model, may overfit

&nbsp;

**3\) THRESHOLD DECISION /TUNING:** Default threshold is 0.5. If prob/ y is greater than 0.5, we classify it as 1, else 0\. We change the threshold according to requirements. For example, in cancer patient case, threshold is 0.1 for example, so even a patient with 10% chance is diagnosed with cancer. With this many patients who does not have cancer will be identified as cancer, but very less or no orignal cancer patient will not remain undiagnosed.

&nbsp;

#### **Business & Medical Scenarios**

| Scenario | Threshold Direction | Why |
| ----- | ----- | ----- |
| 🏥 Cancer screening | **Lower (0.2–0.3)** | Missing cancer is catastrophic. Accept false alarms |
| 💳 Fraud detection | **Lower (0.3)** | Flag more transactions, investigate manually |
| 📧 Spam filter | **Higher (0.7)** | Missing a real email (FN) is worse than spam slipping through |
| 🏦 Loan approval | **Higher (0.7)** | Be conservative — only approve high-confidence good loans&nbsp; |

&nbsp;

**DECISION TREES**: Classify data on basis given features. It is like nested if-else/Tree. It is a greedy approach and uses recursion. For e.g if age\> 50, Property\<100000, then Pension=true, if age \>50, Property\>100000, Pension \= No.&nbsp;

From all the given features, algo checks which feature gives the best result. Then split into branches based on this feature. The way we check the best result is a probability function: Entropy or Geni and Information Gain. We check the Entropy of class/dataset, then split them into further classes/sub dataset i.e left and right child and calculate entropy for both childs and add them, then subtract it from parents entropy. This result is information gain. Similarly we do this procedure on all features. The feature with greatest info gain is selected for splitting and data is split based on this child.&nbsp;

&nbsp;

We try to minimize Entrop/Gini. Both are similar. We generally use Gini as it is fast, but we use entropy when we need that result changes on slightest change in data. Entropy is preferred in information-theoretic contexts or when rare class sensitivity matters. Gini Less sensitive to  small probability changes while entropy is More sensitive rare classes and small shifts.

The algo tries to minimize both. The value of gini is 0-0.5. 0 means best. The class contains all data belonging to same class.  0.5 means worst. It means data is equally split between both classes.  The value of entropy is 0-1. 0 means best. The class contains all data belonging to same class.  1 means worst. It means data is equally split between both classes.&nbsp;&nbsp;

&nbsp;

TUNING: Max Depth control depth of tree. Optimal i 3-5.&nbsp;

Min sample split: A node is only allowed to split if it currently have this number: min sampel split.&nbsp;

Min sample\_leaf: A leaf node should have at least these samples.

&nbsp;

Post-Pruning: Cost-Complexity Pruning : Post-pruning **grows the full tree first, then cuts it back.** sklearn implements this via `ccp_alpha` . Every branch has a **complexity cost** — how much accuracy it sacrifices relative to its contribution. If a branch's cost exceeds `ccp_alpha`, it gets cut. By cut it means, we remove that node and add its result to its parent.

&nbsp;

&nbsp;

### **Decision Trees: Core Architecture & Mechanisms**

* **Architecture:** A non-parametric supervised learning algorithm that recursively partitions data based on feature thresholds. It functions as a series of nested if-else statements represented as a directed acyclic graph.  
* **Execution Strategy:** Utilizes a greedy, top-down recursive approach. At each node, the algorithm evaluates all available features to find the optimal split that maximizes class separation (purity).  
* **Example Logic:** if Age \> 50 AND Property\_Value \< 100000 \-\> Pension \= True

### **Splitting Criteria & Mathematical Metrics**

To determine the optimal split, the algorithm calculates the impurity of the parent node, then subtracts the *weighted average* impurity of the proposed child nodes. This difference is **Information Gain**. The algorithm permanently locks in the feature threshold that maximizes this Information Gain.

**Gini Impurity vs. Entropy**

Both metrics measure node impurity (chaos), and the objective function is to minimize them.

| Metric | Range (Binary Classification) | Computational Cost | Behavior & Sensitivity |
| :---- | :---- | :---- | :---- |
| **Gini Impurity** | 0.0 (Pure) to 0.5 (Mixed) | Low (Squares only) | Default in scikit-learn. Less sensitive to minor shifts in class probabilities. |
| **Entropy** | 0.0 (Pure) to 1.0 (Mixed)\* | High (Logarithms) | Penalizes highly mixed nodes more severely. Preferred when sensitivity to rare classes is required. |

*\> Note: Entropy can technically exceed 1.0 for multi-class problems (e.g., 3 classes evenly split yield an entropy of \~1.58), but for binary classification, it maxes out at 1.0.*

### **Tree Regularization (Pre-Pruning)**

Without constraints, Decision Trees will branch infinitely until all leaves are pure (Gini \= 0), guaranteeing severe overfitting on the training data. This is controlled via hyperparameter constraints:

* max\_depth: A hard structural limit on how deep the tree can grow. Optimal values are typically 3 to 5 for general use to maintain generalizability and interpretability.  
* min\_samples\_split: The absolute minimum number of samples a node must contain before the algorithm is legally allowed to attempt a split.  
* min\_samples\_leaf: The minimum number of samples that must exist in a resulting leaf node. If a proposed split would leave a child node with fewer samples than this threshold, the split is aborted.

### **Cost-Complexity Pruning (Post-Pruning)**

* **Mechanism:** The algorithm builds the full, overfitted tree first, then retroactively cuts branches bottom-up based on a complexity penalty.  
* **Implementation (ccp\_alpha):** Every sub-tree has a complexity cost — how much training accuracy it contributes relative to its size (number of leaves). If a branch's accuracy contribution falls below the ccp\_alpha threshold, the branch is collapsed. The node stops acting as a decision node and becomes a terminal leaf outputting the majority class.

&nbsp;

**CROSS VALIDATION**: It is applied on small-medium datasets to rule out luck. E.g, if you have 100 samples, among which 80 are True and 20 are False and in train test split all 20 False get into Test portion, your model does not learn and perform well on training. So we split data into 5 different sets/folds. Fold 1 has data with first twenty examples in Test and other 80 in train. Fold 2 has 20-40 in test and rest in Train and so on. This way we remove the chance of model being lucky.&nbsp;

&nbsp;

&nbsp;&nbsp;**1\. Core Purpose: The "Why"**

* **Destroys "Data Luck":** A single Train/Test split is vulnerable to high variance—a lucky or unlucky split can falsely flatter or doom a model.  
* **Evaluation Tool, Not a Final Artifact:** The $K$ models trained during CV are **diagnostic throwaways**. Once optimal hyperparameters and architecture stability are verified, all $K$ models are discarded. A single final model is then trained on **100% of the available data** for deployment.  
* **100% Data Audit:** Over $K$ iterations, every single sample acts as unseen test data exactly once.

**2\. Splitting Architectures & Taxonomy**&nbsp;

&nbsp;

| Strategy | scikit-learn Class | When to Use | Failure Mode Prevented |
| :---- | :---- | :---- | :---- |
| **Standard K-Fold** | KFold | Continuous targets (Regression) or massive, perfectly balanced data. | Arbitrary split bias on uniform data. |
| **Stratified K-Fold** | StratifiedKFold | Classification tasks, especially with imbalanced classes. | Rare classes vanishing entirely from test folds. |
| **Group K-Fold** | GroupKFold | Grouped/clustered data (e.g., multiple sessions/records per user ID). | Identity/metadata memorization bleeding across train and test sets. |
| **Stratified Group K-Fold** | StratifiedGroupKFold | Imbalanced classification tasks with grouped entities. | Solves both identity bleed and class disappearance simultaneously. |
| **Time Series Split** | TimeSeriesSplit | Sequential, chronological, or temporal data (logs, finance, sensors). | Lookahead bias ("time travel" predicting the past using the future). |
| **Leave-One-Out (LOOCV)** | LeaveOneOut | Very small datasets ($N \< 50$). | Starving the model of training samples (trades for high variance/compute). |

&nbsp;

## **3\. Production Scenarios & Edge Cases**

* **Massive Datasets ($N \\ge 10^6$ rows):**  
  * The Law of Large Numbers applies; sampling variance vanishes.  
  * Running $K$-Fold CV is computationally wasteful. Use a fixed, single Train/Validation/Test split (e.g., 80/10/10).  
* **Extreme Micro-Datasets ($N \< 100$ rows):**  
  * Use Repeated $K$-Fold (e.g., 5-Fold repeated 10 times with different random seeds) or LOOCV to squeeze utility out of limited samples.  
* **Temporal / Logged Data:**  
  * Shuffling is strictly forbidden. Use rolling-origin forward chaining (Train on Jan, predict on Feb. Then train on Jan \+ Feb, predict on March )where the test window is always chronologically subsequent to the training window.

&nbsp;

## **4\. The Critical Leakage Rule: Pipeline Integration**

* **The Anti-Pattern:** Calculating scaling parameters, imputation statistics, or vocabulary mappings across the entire dataset before the CV split causes data leakage. Offline validation metrics will look artificially inflated.  
* **The Production Standard:** Preprocessing steps must be isolated inside a `Pipeline` or `ColumnTransformer`. `scikit-learn` fits transformers strictly on the training folds and only transforms the validation/test folds.

&nbsp;

**FEATURE SCALING:**  Machine learning models are essentially advanced calculators. If you feed them 'Age' (18 to 80\) and 'Salary' ($30,000 to $150,000), the algorithm will mathematically assume that 'Salary' is thousands of times more important simply because the raw numbers are bigger.

Feature scaling neutralizes this magnitude bias, putting all features on a mathematically level playing field.

&nbsp;

| Technique | Mathematical Action | Best Use Case | Vulnerability |
| :---- | :---- | :---- | :---- |
| Standardization (StandardScaler) | Centers the mean to 0 and scales variance to a standard deviation of 1\. | The Industry Default. Used for Neural Networks, Logistic Regression, and SVMs. | Assumes data is somewhat normally distributed. |
| Normalization (MinMaxScaler) | Squashes all values strictly into a bounding box, usually exactly between 0 and 1\. | Image processing (pixel intensities 0-255), or algorithms requiring positive inputs (e.g., Naive Bayes). | Extremely sensitive to outliers. A single massive outlier will squash 99% of your data into a tiny 0.001 range. |
| Robust Scaling (RobustScaler) | Uses the Interquartile Range (IQR) and median instead of the mean and variance. | Financial data, real estate, or any dataset riddled with extreme, un-removable outliers. | Ignores the actual magnitude of extreme edge cases. |
| Max Absolute Scaling (MaxAbsScaler) | Scales based on the maximum absolute value, keeping data between \-1 and 1 without shifting the center. | NLP & RAG Pipelines. Specifically designed for sparse matrices (like TF-IDF vectors) because it doesn't destroy the zeros. | Doesn't center the data, so it won't fix skewed distributions. |

&nbsp;

**Handling Missing Data** In traditional software engineering (like C\# or Node.js), a `null` is simply an empty state that you can bypass with a null-conditional operator (`?.`).

In Machine Learning, a `null` or `NaN` is a mathematical landmine. Algorithms perform matrix multiplication; if a single cell in a 10,000-cell matrix is `NaN`, the underlying linear algebra collapses, and the entire training pipeline crashes. You must fill those holes (imputation) or remove them (deletion).

## **1\. The Theory: The 3 Types of Missing Data**

Before you write a line of code, you must diagnose *why* the data is missing. Treating all missing data the same is a critical architectural flaw.

1. **MCAR** (Missing Completely At Random): The missingness has absolutely no relationship to any other data.  
   * *Example:* A temperature sensor randomly loses battery power for 5 minutes.  
   * *Solution:* Safe to drop the rows or use basic mean imputation.  
2. **MAR** (Missing At Random): The missingness is related to *other, known* variables in your dataset.  
   * *Example:* Men are mathematically less likely to fill out the "Depression History" section of a survey than women. The missingness depends on 'Gender'.  
   * *Solution:* Advanced imputation (like `KNNImputer`) works perfectly here because it can look at 'Gender' to guess the missing value.  
3. **MNAR** (Missing Not At Random): The missingness depends on the *missing value itself*. This is the most dangerous type.  
   * *Example:* People with massive credit card debt leave the "Total Debt" field blank.  
   * *Solution:* You cannot just impute the average debt, because the fact that they skipped the question is a huge predictive signal\! You must use an Indicator Variable (explained below).

&nbsp;

| Technique | Mechanism | Best Use Case | Engineering Flaw |
| :---- | :---- | :---- | :---- |
| **Row Deletion (Listwise)** | Drops any row with a NaN. | Huge datasets where missingness is $\< 2\\%$ and purely MCAR. | Destroys valuable data. If 5% of rows are missing across 20 different columns, you might accidentally delete 60% of your dataset. |
| **Simple Imputation (Mean/Median)** | Fills gaps with a single static metric. | **Median:** Skewed data (like Income/House Prices). **Mean:** Perfectly normal data. | Destroys the variance of the column and pulls everything artificially toward the center. |
| **KNN Imputation** | Looks at the $K$ most similar rows (based on all other columns) and averages their values to fill the gap. | Datasets where features are highly correlated (e.g., guessing someone's missing weight based on their height, age, and gender). | Computationally expensive. Has to calculate geometric distances across the entire dataset for every single missing cell. |
| **Iterative Imputation (MICE)** | Treats the missing column as a target variable and trains a mini-Machine Learning model (like a Ridge regressor) to predict the missing values. | The absolute gold standard for complex, multidimensional datasets. | Slowest method. Prone to overfitting if the dataset is small. |

## **3\. The "Missing Indicator" (The Pro-Tip)**

When dealing with MNAR (Missing Not At Random), the *absence* of data is a feature itself.

If you just impute the median salary for a blank field, the model forgets the user skipped the question. By adding a **Missing Indicator**, you create a new binary column (`Salary_is_missing`: `1` or `0`).

* The model now gets the imputed median salary, *plus* the mathematical knowledge that the user originally hid that information.

## **4\. Edge Cases & Exceptions**

* **Time-Series Data (The "Mean" Trap):** If you are predicting stock prices and Wednesday's price is missing, you absolutely cannot use the average price of the entire year. Time-series data relies on temporal flow.  
  * *The Fix:* You must use **Interpolation** (drawing a straight line between Tuesday and Thursday) or **Forward-Fill** (`ffill`), which carries Tuesday's price forward into Wednesday.  
* **Categorical Data:** You cannot calculate the "mean" of strings like "Lahore" and "Karachi".  
  * *The Fix:* Use `SimpleImputer(strategy='most_frequent')` (Mode), or replace the `NaN` with the explicit string `"Missing"` so the model treats "Missing" as its own distinct category.

&nbsp;

**Encoding Categorical Features;**&nbsp;

**1\. Core Architectures & Use Cases**&nbsp;

| Technique | Mechanism | Best Use Case | Engineering Flaw |
| :---- | :---- | :---- | :---- |
| Ordinal Encoding | Converts categories directly to integers based on a hierarchy (e.g., Junior=1, Mid=2, Senior=3). | Ordinal Data: Data with a strict logical ranking (Education Level, Survey Ratings like "Poor" to "Excellent"). | The Magnitude Trap: If used on non-ranked data (like Colors: Red=1, Blue=2), the model assumes Blue is mathematically "twice as large" as Red. |
| One-Hot Encoding (OHE) | Destroys the original column and creates a brand new binary column (0 or 1) for *every* unique category. | Nominal Data: Data with zero inherent order (Cities, Colors, Departments). | The Curse of Dimensionality: If you use OHE on a column with 10,000 unique values, you add 10,000 columns to your matrix, risking an Out-Of-Memory (OOM) crash. |
| Target Encoding (Mean Encoding) | Replaces the category string with the *average value of the target variable* for that category. | High Cardinality Data: Variables with thousands of unique strings (Zip Codes, Product IDs, IP Addresses). | Target Leakage: Because it uses the target variable (y) to calculate the encoding, it is incredibly prone to overfitting if not carefully cross-validated. |

&nbsp;

## **2\. The Production Bottlenecks**

### **Bottleneck A: The Dummy Variable Trap (Multicollinearity)**

When you One-Hot Encode three cities (Lahore, Karachi, Islamabad), you create perfect multicollinearity. If a row is not Lahore (`0`) and not Karachi (`0`), it *must* be Islamabad (`1`).

* **The Crash:** Linear models (Logistic Regression, Linear Regression) try to calculate independent weights for every column by inverting a matrix. Perfect multicollinearity makes the matrix non-invertible, causing the math to break down and coefficients to explode to infinity.  
* **The Fix:** You must drop one column using `OneHotEncoder(drop='first')`. The dropped column acts as the baseline intercept. *(Note: Tree-based models are completely immune to this trap).*

### **Bottleneck B: Unseen Production Data (The "Silent Failure")**

You train your model on users from 3 cities. You deploy the model to an ASP.NET Core API. A month later, a user signs up from **Peshawar**.

* **The Crash:** If you used standard pandas `pd.get_dummies()`, your pipeline has no idea what "Peshawar" is and throws a `KeyError`, crashing the API endpoint.  
* **The Fix:** Never use pandas for production encoding. You must use `scikit-learn`'s `OneHotEncoder(handle_unknown='ignore')`. When "Peshawar" arrives, the encoder safely ignores it, mapping it to a row of all zeros (`[0, 0, 0]`) so the pipeline continues running smoothly.

## **3\. High Cardinality & Advanced Architectures**

What happens when you need to encode **Zip Codes**? There are over 40,000 zip codes. If you One-Hot Encode them, you create a matrix with 40,000 extra columns. The matrix becomes 99.9% zeros (a highly sparse matrix). Neural networks and linear models will choke, and training time will skyrocket.

**The Engineering Solutions for High Cardinality:**

1. **Target Encoding:** Replace "Zip Code 54000" with the average house price in that zip code. You compress 40,000 columns down into exactly 1 highly predictive numerical column.  
2. **Frequency Encoding:** Replace the Zip Code with the percentage of times it appears in the dataset. Useful if the *rarity* of a category is the main predictive signal (e.g., fraud detection).  
3. **Dimensionality Reduction (Embeddings):** In Deep Learning, you pass the Zip Codes through an `Embedding Layer` (exactly like word embeddings in LLMs), which compresses the 40,000 categories into a dense 32-dimensional mathematical vector space.

&nbsp;

[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnAAAACrCAYAAADmdaFmAAA/R0lEQVR4Xu2d+bNWxbX37z9xq27VrbrlrdRNpfImP8Y3plJvUtGUWjHGqFFjjEbRGAngCCpgIuDA5BgRFRFEnFBQBEVRFBGUeZ7neVbzH+zXT3vXznrW3vs55+E853D24fvDpzi7u3fv3j2s/vbq3g//9u///u+ZEEIIIYSoD/8WA7rCxRf/Ojv33F8Uws80LrnkN9kPf/jDQvipQD5nnXVWIby7ufLKK07Lc0Vr/OScc7ILL7ygEN5bkX34lqlTJme/PO+8Qng7WPTpJ9ncObML4UKIvkWHAm7o0HuyqVNfaMrZZ/8ou+3WQdnXX53IJj33TCGPdjFq5Ihs187tadKKcT0Fz//44w9Ly4Dguf/vf8sOHtib3XRjv0J8q0ybNjU7dHBfEsYxrrvgvXjutq2bevS5onUuuuhX2bGjh7N5780txPU2sA8rVyztVvsA2Iiq8VmGjecY3p3cOmhgtn37lkL4go8+yJYtXZJdffXvC3HNeGHypJSnXV//p+tK8xdC9C06FHBMDvv37c6v//n1yXzCYAW54hujzESPeCGuOw10bxdwgMcM0VUl4FqZLAYO/Gt2+ND+bhVSvMff7hveEMZzuyLg3p/3bvbUU08WwkX7OX6sHgIO+zD/g3ndah8gCjie26wvnoqAazV95MjhA9l9w4fl1++9OyfbsnlDfr11y8bsy5PHmpb75pv/nL0x47Xs5ImjqQ888MDIhvh+N1yfzZr5RuE+IUTfoVMC7q23ZubXXsDB2DGj84m+uwVcXagScEwqCN4YXgV5dLcH7tFHxhWMP8/tioBjglI/6BnqIuCAcvZ0v7jrzjva+sxWx3CE3YpNG9c1bCWzSJs4cUJ+zXikXRmD8X7j2mv/mDxvq1YtLxVwsHPHtuzyyy4rhAsh+gYdCrghQwYnY2HXUcDhhcMoWdzzk57NduzYmrZTV65cVjAgiEFEyYnjR7LPFi0sPA+GDbs3275tc3b0yMHs2NFDWf/+t6RwDDErZi8s7r57SLZ50/pvjNiRbN/eXWl1vGf3juQJGz9ubHZg/57snnuGZAsXLkjlIw1x06dPS9dsQcWzXjyP/Fjdvv32rPysCsab50dx8/prr6RtU96Zdy8TcJQBw02exNvqmnclnDCgnHZPFHCWBvz7cz/1uXfPzvxdKDMre8QUBp5yx3N5rOCJo57Jc/WqFflzeUdW+dQP9fTMxKcb7qXcTDxWZrbzyH/y88/9b70eSnFs5/j7rA3pF5TP8qW8tBV5sv3zyPhxKZzzPPbOF5x/fgobNHBAXr88mzjK6/spPPnk46lOKAv5XnrpbxvKTzjtQZyvm2btT/l5FuX/6svjeflv7HdDuof6jPl5aKtYb4STnn5MHfBM0lD30Yty5x23Z2vXrMr7LmWsEnB+DH2+5LM8nPeIfRavD3Vq7WNp2c6jncjHwqif5cs+T+9A/+LdLc76HeWnLjjrZXVBObEPjz06Pj0TLxM2wpeZspEvdYl9iG1GOYD3Mbvg77X2YczQF8mrqi/68cw170relJN2onzUzy23/CXFV41h3pk2oy6Io278czx33H5bob2oK3sGIO7sOfH+CHlVCThsgo0jIUTfo0MBF4kCLsbNeeft/JoJwowjwoL7vBF9+RsR9dJLLxbywTjaBG338TdGzm8pYji9mGJSZcK2iZyPCUjPlqDlheH3WxNMKJY/BhyDvmDB/DyeON7LrsnPP5P44cOG5vHgy+8hLK7eo9cSAcq5Q0tPXvYs3s3igHMvfoLnwwNbdTNp+XYaPPiu0jKRJhp/0vEszi1ZmC9nrBPKzPPK0kasDdesXpnnH9+TSY7J0O5hIluyeFH62yY3xKYtHMq2tEjn28kmd/7m+a+8/FKeljq1dumo/Sk/EyPltzCEkl+MUC6e4evP8opt7esN4Yoos2v6JgLCrgcM6J+e4xccVR446suPIb8w4JlVfdbax8dNeeH5hjqINgAhy5iiHmO/oz9aOQjHPpigY4GAjfBjiX5n92IfeH9sBO9DGS2O9yl7b8rv2536ruqLYOPZrnk3L3bJx4/nOIapW/qMP7fGeIqLPIMy+zGMrSpLy8LAt30V5Fcl4KhLK+sTjz+a3k0fNwjRd2i7gPPGEsNkxpHJLE4MGJ2y1SoTAt4hu64SNNGYgjdm0fhCFCz8be/DRBFXrVZuE4Xm7SFvzpng0YsevPhMo6y8X3y+OJ3bsWtfPv++TKJ4nvy9H334QRI7PgzPxcgR96d/mTzNO4KoKPvqLdaHPTdOKr5taQ/K7fOg3svSRuydrrnmD3kYZZs9e1a+rWRpLJ6JyA5lc3YIDxRCgXJQxnVrVxeew5eZlGnChH/kYQgG/o0ejyt+d3nyCvF3R+1P2Xbv2tFQ/ngPMPlGryXljW3t64069+9tW2l2jfc6tlWVgGNcMYasb9q702f5mKCqz8a6B9rSCzi8ov7drL19v7O40aMfyvsd5fT9grbz/Yx69mLc3p93AeyCfx8vhIzYdzsScDae7brMvjUTcPSPuICgP5uojc/jXm8bYnmN2PZVNBNw3v4iMKlf2iimE0LUkx4TcNzjJ4FmPP7YI0kYkZ6J8KqrrkzhXtBwjaHEeDFJcM3k5D0UrQo4m0Cj+MLo2j1ewM18c0bpO5XlAdH4A1toeIPYOmMiwNMQBRwChXrwH07EuoiQL3XBNhBlZGsoTtoQ68PyjpNKbFvyZ9uNclPmVgVcLDdlW/zZpyk/6tWLffPE4FlEvDFp0j/YcsRL5T1mnhdfnJLem/JQDwiJsnczyvoL+PaP91fdUwVtTb1ZW3dWwCEgEbKx3FUCDvwYsu1x6rasbVoRcObNJN3SLxY3tLf1O8LguWcn5v2uIwHXrN8AdsHy5X3MLnhi+3S3gIvv1Cwf8J5RwGuOtzkuxNol4PBWx3AhRN+gxwQcrns/CXQGO3uzccPafPKNkz/n3TCATPJ8yeXP3JVNrlGweAGHsSszhhhsW+17AVclSuMzjWj8LS0Hke3al8/eF28Rkz71YZMhYhWxE899RZhsETpM4mUegVgf9twocnzb4kWiXJTB8uiKgKOMny78OE32/B1FhG3N8cEM6agDEz9spUXvl4e0eIxIi7C46aYbG7bZPXgAO2r/WDd4+swbGPOLUG+0dVW9NRNwVidM+D7PZgIOeJaJWMYQaRkrMV0rAo4tQ7aNrS+WtTdtQr/zX1NGsVMm4BCYPp8I78M4IC12IcbH9uluAUd5Y3rLp+xdsIO+f1V56x566IEuCzjKXbbDIYToG5ySgONnImK4xVUJOIQVWyscavb3zHj91fwsk+G/egWECoY7Tv54X8qEktGqgEMgsB3mJwa27Pw5JS/gbHuNL90snkktPtOIxp88opigbqOAs/fl51yYEDmAzjVbvgg7L1qpX34WhEPiJhSAPMvOv8T6sOc2E3DE+bNKlLkrAi4KhLJzWHis/Lva5F+2HWjx/ss+rrkfwYWYjb+TZf2wo/Yvqxvu8T+1A3xAYduy9nzui23dWQEHbJnjbfTvWyXg4kKBa/oDfZYFT1Wfje1DHFuxvn2iYLX2pryx31E+63cdCTjsA/3bl5u+QNtQfm8XuMYu+LQQ26e7BZydN/VnChHKtuiMz6Nu4pYrY8mf+eQ+xK+dq+RjjapFSjMBx3lK+7mSH/7whymtfTQjhKg/pyTg1q9bUwi3OM5q2TUrdc4L2fXDDz+YDDTeEwwLvwsVBR0wGdoEc91116ZtGiZXm9j5OIE4BAyGjh+/NCY+PSE3nPYRgz/vhPF84onH8mv+NoNq3hpWxBg8wljBkq+lJz8mfysDHxEwkSAWOF+CcUcg2KF7D8abyWDUqG/PQfFOpGVSJo578JZY+eL7mufLJjm8VdQ5EwBfJzJRmReSs0gYbN6D90L0jRxZPP/C5PrJJx81GHae698ReI4JR7avKDf3mIfHfzFHHFudZT9IGt8JTDjzfj//+c9Sn+Ed/ZYxHgoEjM+LCR0RH58BJthsm+3ee+/OhQTtyftw7o4fokZIWVxH7V9WN3iEeOdZs95MX21TVrYv/ZlF2pp6o62pN2trX2/k64UrkzbxJtj4AMj6FuWgPbim//m6AvqhH0OMRVso2T2MVeuzJlJMPNB3eAZjnXupL35yhrLwN55E+hnprG98Kzi/7Xc8h7SIG+t31LO3D3gTydvOE2IfEB28H39jH6gjbARl9CKZ+7AL/p0htg9n9ar6Ith4tuu4QKXOKAMfAXEdxzDQRtbXeGe27uN5VYP4+PMe9t52XpX6o0yU2foNH3vYOUaPfZFsY9Ow+6zNaWfSlXkFhRD1pGUB1w6YXDk4HsM9bGdhLOPE5GHLjfNv5jmADevXdurrrY5gsvSehGYw2ZPeJv1mZea9vHcOg84EZvdCR+8d4d6yeyzvjuqaNL78nYG8/T28l38+z+xs/RnUi/f8+HJzHctHmPdGlUF+Ve9GeLO4VstPPfIOzf67KKtru4711hl4J6un+DM9HutrZfnHPhu9xsSZsCIfe55B21i+/OvrijYhr476XRU8y5fFQ3ir7XIqfbEZcQwbPCfWUxkINIRXDOf+snyBc7Bl2/5lUP8sFGK7//SnPy2kFULUl9Mi4NpB2VetYF6LGC6EqCYKONF9ILpb+bgAr2j8PcBm4PGMRwSEEH2P2gq4eD7IYCuhLFwIUY0EXM/CFmvVNquH86x8qNWRp9mDd49t2RguhOhb1FbAAds/9uv1nDkCTUJCdB7+dwMEAgLOf0AhhBCid1NrASeEEEIIcSYiASeEEEIIUTMk4IQQQgghaoYEnBBCCCFEzZCAE0IIIYSoGRJwQgghhBA1QwJOCCGEEKJmSMAJIYQQQtQMCTghhBBCiJohASeEEEIIUTMk4IQQQgghaoYEnBBCCCFEzeiUgPuv//qv7Pvf/376z+OFEEIIIcTppUMBh3CD73znO9l///d/CyGEEEKI00ylgEOwIdziDUIIIYQQ4vRSKeB+8IMfFBILIYQQQojTT6WA+5//+Z9CYiGEEEIIcfopFXDf/e53CwmFEEIIIUTvoFTA6eybEEIIIUTvpVTA8XlqTCiEEEIIIXoHEnBCCCGEEDVDAk4IIYQQomZIwAkhhBBC1IxaCbhr//jHQpgQdWTF8qXZ50s+y37wg/9TiPMMGTI4e+SRcYVwIYQQZza1EHD9+t2QfbZoYbZ61fJCnBB1A9F2YP+e7M47by/ElbF/3+5syOC7CuHi9DDtxSnZ9OnTcq6//k+FNO3kyiuuyL7+6kQ27725hTjPieNHsn17dxXChRB9k1oIOCa7f359UgJO9AnGjH44mzPn7UJ4Fbt3bc/Wr1uTnXvuuYU40fO8//572aGD+5NN4t/uFnDjxo5Jz9q+bXMhzkMaiOFCiL5JLQQcIN4k4ETdeeyxR7KTJ452uHXquffeu7Mjhw9kmzdvKMS1A8qC9wahEOO6yvFjh7MPPnivEF53qKtW64x6Rvy1ck8Z3dleQoj60GUB98CokWkVGlmyeFGehr993OzZb2XPPvN0wzWT1Ib1a5Nhgs2b1jc8xws4tlQxhEePHMzjR464P8VPnTI5D/vxj/9v2n5i+wFIE8svRE/y6acfp77ow9auWVUYP/CXv9yc4n/+85+lsYEYivkZO7ZvSfnigZk7Z3b2m99cnMLx2u3etaPBe2Njh/Fy++23ZatXr0j3EkZZLr30t2kccd+jj47P82Z7bsKEf6Q8GFsIyoMH9ub5znzj9XTNs3j+u3PfSeVhPO/dszPZiljuuhIF3O9+d3l6f9590KCBKY53f+Xll5LgivVM+1LP2DKuTeSSlnb58uSx3JZZmmbtZe3h+whtQFtTDvLz9s9sqOVJPOmsfYUQvZ8uCzgM2N13D0l/c06H7c6FCxcUPAyjRo5IxuW1V1/Ow/7x5ONZ//63pL8xHmvWrMzv+3D++9kXXyzO03oBx6S0bu3qhu2Cu+68Izt86EBu9GbMeC376svjDWUgPQbPhwnRTqZNm5o8bEyQTKr0+cWffZr6Of2WPupFDzDp27ggDf0U8RTzLdseY1wdO3ooF2xggou/GU9x+40x4I8kDBzw14JHh/KQxkQk8OEFYXZ2j3sQIha/4KP5hWdxfSZ44GjrTz7+KNXHww8/mMJoc95/9tuz0jX1TLyvZ2tvX0f0EcLMllmaZu0Ftvjlb9od+zdp0rN5PNcsaG+77dbchrI9j4Ajnr5EGhbXPl8hRO+kywJuyJDByUPA35zTwdBcc80fCulIw8SBJ4G/MTBLly7J41nde68dBs1vmcYtVP72E5oZRzN6GCa2nXwZ4oQjRDtB7NCPmSQtjEkd74Zd0wfjWSY80PRf/mbhQZr45Sn9mv4eF0YscqKwQzAQ9qfrrk3XUVTZ2GkmCOx5HKC3MFuEzXj91XQdxxNjNj4ripO+QhRwwHvy7lZnJp7t/csEHMQ6Io0XcJamWXuBF3AjRvw99cU/33RjHs81+Vj7RZv6+99fVXiuEKL30mUBZzDhdPSlFMIOA8Gkxgofj5vF4UHgi64tmzdmy5d9kVaHUbDF62YCLk4kQnQ3JmD8ZGwTvYmpMgFnbNu6uSCaDBNUMdwmex8WxUUcC6cq4Gwr195PAq65gAP//j0p4MrKEts92lTSxucKIXovbRFwCDO2bTrzpRwGAu8YZ9y8e5/VIYbL3Pdd9cARF7eqhOhOTMCw/W9hUcDRR8u8wOatiV5jA6+J7+9GMwGHF4brKKriRF4mCMoEHH+Tj20JSsD1bgG3c8fW5FWzeGt3O5oSbaoEnBD1oi0Cji0iv03E1qg3DB68bxgJ/zMKZjj8xMffUbDFa+6xLSUmKw71mvFBIBI/9N578nvY7t20cV2hTEK0A9tC9SKMseDPYtLvo+DiTBLbrpwxswUQixvbVjXhxNmo+Ey2XPF821lSYPL2H0rwvEMH9+XXjIEoCDi3x5lUS1Mm4OxZVkYvGICzfmUCjg837Lqv0BUB5+vZ0qxauSy/xo5FIdVRe4FvD85gRjtLH+BsJXFcR5sqASdEveiygOPDBQY9255cX3TRr5KYq1p12zka+/DBIA+Mz1uz3sxWrliW/t66ZVMyKkwYGzesTWfoOCxM+lkz30iTCV9k4Z3gIDfPZSLEAI0fNzYPY8JicsToLfxkQaFMQrSLSc89k/ouYo6+Sv+j71k8/R4hZudGwT4OQMhxTd/Fe2xCwMaMPzNqkIbFCv+rgwkrRIJ/pgkCxgwfGlA+G28cXbCJm3Fi48sEnB3IZ6FEGfhIyfLly1LS8LUpMN645p1v/vNNKQ3XlAe7EMteV6gLsz98eWphiz79JNu1c1vubbV6JdyuEXjU8333Dc/zs6/ksWMIKvsi1M5S2kcM9gFWWXsB9s0Laq5tMUEetJ+drSyzqbcOGpjytXcSQvRuuizgbNskwldzMS3YxwzxMDaGz+7l8/bhw4amvzFIdvgW7JfG2ba1r+kwfmPHjE4TBWfonnt2YkqDx81+WgE4n9fRFq8QXYWfa2ASpu+9/torDZMs/Z6vtBFlFmaCKmJjBK919PZ4+AkLnkcaxCHjx3+V+s47b6fJnDwRBYgEysbflo7JnXiEI948E3B4DxFm3M8RiQED+uf5Tp48Kb8PkTj/g3npb8pgP27LNp6NUdLHstcR+0rUeOyxR3OvptUFP5liwhn4qQ/u5YiIhZnXFDtmIo46xo7Rntgy8+raPfZVcGwv8+yCed34SRDaGBFHG/qfEYk2FZtpwhFsQS6E6L10WcAJIVqHSZX/kSGGRxBxfNxQ9mV3d1K2hSqEEKL3IAEnxGmAbSq2PjsSZk/944nT8v+gSsAJIUTvRgJOCFHAvpbta/+DghBC9BUk4IQQQgghaoYEnBBCCCFEzZCAE0IIIYSoGRJwQgghhBA1QwJOCCGEEKJmSMAJIYQQQtQMCTghhBBCiJohASeEEEIIUTMqBdyPf/xjIYQQQgjRC6kUcDFMCCGEEEL0DiTghBBCCCFqhgScEEIIIUTNkIATQgghhKgZEnBCCCGEEDVDAk4IIYQQomZIwAkhhBBC1AwJOCGEEEKImiEBJ4QQQghRMyTgupn7hg/L7rj9tkJ4nbj00t9mPznnnEK4EEIIIU4PXRZwQ4fek02d+kLimYlPZ78877w8bvLzz+VxpIv3tpNPF36cnTxxtBB+Otm+fUs2fNjQQvjp5rFHx2ejRo4ohNN2X3y+ONVl7APr1q7OXnxxSuEe0XUuvvjX2cvTpxXCPbTHU089mS1ZvCibMOEfhXjRc2DnyuxdZNDAAbn9i3GdgUXToYP7shUrlhbiyqiygeTz8ccflo75Mxnabv++3al+9+7ZmR09crDSxp111lnZls0bsoMH9hbiuoNbbvlLet7ll11WiDMGD77rlPtWxPrOvPfmFuK6i6q6Fp2nLQLu669OZP/8+mQ26blnGgzajNdfTQYI49GdAu7cc3+RlyHGnS4wmp8tWpgGfow73WC0aCsfhnHfs3tHMgiIhM2b1mdXXXVlHo94wMjdOmhgIT/RNd57d062beumQriBMd++bXO2fNnn2Yfz38+OHzuc3XXnHYV0omd4YfKkbOuWjcnmvPTSi4V4YPyvXbMqtdXcObML8Z2BvLFpnRFwzWzgTTf2S3Z46ReLC3FnKrTPsqVLchvX74brs317d33TXkcKaeHOO25PdUs9xrjuAPvM8yZOnFCIi2lieKv4vtNTAo7670y/Fs3psoADOtlXXx7Pzj77Rw3hDAomnpi+O2hXZ24H1MPqVSvSwIhxPcEDD4xMRjuGP/fsxCTQTGxbOEITwebFN2KBNvX3z3xzRnb40P5CvuLU+Nt9w1N7MHE0E3AIZ8aSD+tJYyuKMMZoty9PHivEwedLPkvt1tU2asUD15tsYG+HdkFcx/Bm0A7tFHAI/M62rS0IfNjIEfdnx4629g5VWN/pan/tCBwa2Lw333i90+8uqmmLgMPNu3PHtnTey4ezgly1anl+bVsCrHIQDZytIvzqq3+f3Nd0HlZEB/bvyUaNvD+lBfK44PzzswUL5qdrPEXDht3b8KxovIjftHFdnsfChQvyuPHjxibBcvPNf05uXFshU7633pr5v2U8nK38poOxdchWqN2LsKH8xE+fPq20rjjzduTwgUI4UCbux7g/+eTjhXjjkfHjUh48B+G0+LNPkxcspiujSsAZUcCxLfD+vHcb0vCsKMrJt1Wjd6ZxY78b0uRNH6L/WDhijVV8TA+ItyoBx/Yq/T96co8dPZTt2rm9kF70DIyFd+e+kzwXceEK2Kg577xdmBCxF3jAsQHYgmjHrr32jymOMcqWFuPNT3Rmo0hz4viR3IZCtIHGgAH9s0WfftIw5rE92CD6EYsynw9QLrOdb8x4LYXRR7099vaSsmDT7X6eZ57i5Lm/dVC6z9t00nmbHssN1JGVEXvry4cNpx4Q0n47Do8284nNAbQRZceTTTzlZo4h3MoD3OO3SEn3+muvpHTAe5DOl2/qlMnp+bTVmtUr83DqmvHJHMPcg9DiHdh2J/6ee4bkbUye0bbbHGV2nPcwD6ClJ87qDftA+ZgzeM6Cjz5oyM/o3/+WfA6jTLZoLxNwPJO8eJ6fP4E6tvmJ97R5MParqnJQdgm4rtMWAQcM5I8+bGysjRvW5i5gRB4rCDMUGDG2IfBS4dqn83A/99BhaGAmQML95MVALdvGi8aLzmkTKGdReJ4ZWgYqaRmcrKCt0+JSp0yISAwP96xftybvvGwzYsyoH8rE4KMjx7I8P+nZ0pU5HrCnn34q3cszy86rAEbBBCADjLLu3rWj02dYWhVwCLUpLzzfkAbDgzHznh8mFwZnzE98y4UXXpBtWL82bUGPGzsmGSiMGIsA6u1UBBxHD6K4tnuqtntE98MYo21YuPK3j2N8I0ywA35CRBBg667/03Xpmgna2wDGN2OOBa2lx474iQ4bauIHG4W9snNS0QYanKVCgPgxzwTLIpGyYgd9nIHI8eXD7rGzYB80WVmw6WYvKcsVv7s835Kj3/MvCxHuwybP/2Beg03nOWU2HZtk9pJ7yNPiGFdWfuoYG0b+fjuQZzN2KDNh2HLu5VnYU+7B5gPlok68QOP9EGc8A0w8WTzPY4cJUcScQVsw1xCHsKEM3E/+06ZNTWHMPcQzjyB6aW/io223OcrsOPVMeisv6S+55Dd5X6CvUR/8bfOaz8+g3ng274MgtmM+UcCNHHl/qhPeMc6f/ItTg3e+6KJfZStXLsvLGftVVTkk4NpD2wQcDcYgoSNZGKKOf2/7ZvVFx+EsnMVFbw6dB6HkBzbiBSNnnj2MAwIvPhui8eIwvh8UdEwztBgTJkAOgfo8KI83xvxtnQxvIgPIpydPb1SMMlc74jR65UiHGzzeTzm84beVb0znYcXrV5I8y6/WfNoo4OK1Qfni5ERYM3EoWqeZgKNdohcHCCubrEXPYIskFjhMWv5jJdqGSc63ne1SIOZ9PtgFPOz8zWTn7Scwfs0GYUe9DQVsBR4V/o420EM5bIyz0OA+/zEMC7Z4D3bXe2mYmE0ElZWFOrGyVPVP8iJPb9Pjzg1QL/64BqLQFtLExS/7x44ZnTxGCLiyeog2mfL5+cfmBEtDuyJg/Hzk87D29N5X+gMijvq1c4fka/GxXL5tI1Yes7WWX0wHPI/n+vYsO3POHPbQQw/k19SVneGMAo77q+ZP5izend0GrqkD2rWsX5WVAyTg2kPbBBwgcFjt8DcDgEHO37iNETq2wjGR4cWP7zweOpiFM0irGj0ODoMVxJgxD6dVaBRwUYiwIvNih/T2bAxTdLnbVkd8ZjQWwADDQPv7Wd1GgQQMArw4dt1sgjdGj34o/+KNMnPGwK6jcY6CjWubRDwScD1Ds/alnXxfMKomSNEzeC837WC7D9g929r2Ao5JD3sRRQ/2xkQP9jFOeH6Sx456GwrYJIuvsoHgBRzYtuArL79U+aWjneXF7nJNP7QJv6ws2HQrS7P+yXO9TS/bgsZelo0JEytXXnlFQzi7A3jyqP+yeog2uSMBR7tEm+jzsPb078+imTzIqycFHMya+UaqV7ZUq9qTflZlu6OA88T503aFmM8Q9eaRhc70K5CAaw9tFXAMbhqPBvYrTc6YVYkEo6rzAMKKrVjyjIe5jbLB4c/f+RVEHBzG7NmzUqfEaPG1nz8PgdHAYMXnlsH7xrNiGJfObj+OGjUiGQNc3cDK0w+SjvCTSxlRwCEkbfVqYaw8qT9vKK3eMKIxT9E9W6h4GriXCcqHszUUPbqi5/BjjDOPjCl2HNhSs2MjXsDZmdI4aZEP/YWxRXtGz5Kf5LErzWxotIGeKOCAMW5n1aq+mGcSZhsQu4tNtPCOytJMwBFHXZBn9Ega9Pkye2nHOKJ9o/5snJTVQ6sCjvj40z4+D3NK+HhPTws4oP3YHrbtW9+e2HZsfFwgGFHA4X3D/tt86+dPgzmJHTLqwTs+fL+K5TAk4NpDWwWcfYrN2QU/MM14mXeujGYCjkmQe+MZO48fHKzouMcOrUJnBBwijUEL5Mf+vsVhZJsNWA/vG/f+GTxxC7YKBgUGkgOgzYxkFa0KONzhflsDEG7xkDztixCN+Ylvod9FzwDbCXhBmwnwZgKOPG0B48MZX83Gk+he/BizrUYEBJ4Km7S9gGPiLPvwBCFg26Zl7ewneewKbV7msbLnReFieAHHZM6WpMVxlMQ8R/E+wO6xOPf2oaOyNBNwdqSGM8FVNp3+Hb+CBxMidt7L8FueZfXQqoCjXRDmVVuo1p5VXqaeFHDUiX1oArQndjq2J95fO9YU8QLOPK9V86c/r2hn6Ygv61dl5QAJuPbQVgEHDHJbsfnwhx9+MBk5hBB/cyjVDnQCnafssDbgZkf8NPvRTESjF3AYHQ4SUw6eybOeeOKxFM/hTw5h+g7KQGQFvGL5F/lBUbDf26Kj2pc3s2a9mQ0ZMjgd3uRdY1lIy+D3g5sJnPww8riXGTD2Q42INb99gvjDgFkZEHJmsMybWbWS6gjKxv3+qyIrG881g0U5X31lesO9GFvEXsxTdA36lB8LbHXQRuZ1o98xmdlkx+Fh+kTZylb0DNgSji3Ytdko+0CBtsGjz5kxE++MWdrabApeWg6HWx6c78VukY4Jjh9C5xq7aYtJ/macYkOxtfQbPnoiztvACLbVPlRiQrWPtbi+9967m/5WHbasTExZWbCvlAWbbmXheVVlAeor/nSRx2wSooV6JD/bEeEenssHbdSzfUBg4yHWA/MB9tSPMd7Jf6Bhc4I/d0fdI0BoC7ZMeSZh9qEC/3KN2BkyZHCaFxirlIO5hbzI1/LDMeDLRd68X9k4jnOULZ7ZnYnpTez59vTvYXCf9ScWlZSXj2qoH6sz6tkEHPMn/S7On3a0CJ1AnnyswEcPZf2qrBzw6CPj0ock11zzh0Kc6DxtF3A0ftWqis+66UB0FDojnxyj2lHihBG3Y8fWwn0MWFZdMdwgDwYXefDlEFtZ5MM1nZXzYA8+OCqlYavAvvDhDANnBywfjCvhHu4xo0S92L2AceYLpFge4FBzXE3zHuZW5l3tbBMrcAarfX1mn4xHMGr2haulbQUGJULR8uPatkMp27efu3/7+TerY28oMCCIOl9fomvgxaCurT3of4QxmdI/bOKn33370xGHk8GnD3ZljIqugW2hfRiHZq/sgytLY2d8aVd/9ALvBcKBiY57/c+I2GRp/QFvHe2NDbvuumtTGtKbbQNsKOO0zAZavkzuVl7bKsX+Uka2/ClfMw8xYovFZgy3snibTt7YJgvjpzXKjlxQX1U/guzTmL2kDvxPhWCrqEcW0Iwh+4mRsnrAbll5sF+IbLtm7uEeb9dtO5Dn8Vx7PluK5O2FO8/4VzsfzselnY+mTvgJobJysbvCO4D3oIGfo6w8pCcspqfOmaPsDCLtWSXIqSfLm3zsy10rG+9C2RCOVh9x/qSfUi7O24FtdZf1q7JyYOeszsiT+olpROdou4DrDvBYsacew9sJotOvlgxWbqe6bYgQi1+VdQTG0n8pZLB6s98QOh0gMFn1x3AhhGgVbHr0JAkhWqPXCzjKwjZDd/+foqxcygQcbl77SqxVcLGzEjGXe2dgazUKOAwdK05+kDOm7wlYpZkXUgghugJeIL91LIQ4NXq1gMM1jovVzhXE+HaCd8n2/C2Mw7Zsddjv3ZwKnAsoO8RZBWfi+JLNfoIFOG9QdZagJ0DAdXf9CyH6NnZkgO2zsh9AF0K0Rq8WcEIIIYQQoogEnBBCCCFEzZCAE0IIIYSoGRJwQgghhBA1QwJOCCGEEKJmSMAJIYQQQtQMCTghhBBCiJohASeEEEIIUTMk4IQQQgghaoYEnBBCCCFEzZCAE0IIIYSoGRJwQgghhBA1QwJOCCGEEKJmSMAJIYQQQtQMCTghhBBCiJohASeEEEIIUTMk4IQQQgghaoYEnBBCCCFEzZCAE0IIIYSoGRJwQgghhBA1o20CjnuWLF6U+Mk55xTihTiTmDr1heyLzxdno0c/1OF4emHypJTeGDr0nkIa0Xt4ZuLTqZ3495fnnVeINwYNHJC3aYxrFWzqxx9/mI0aOaIQJ9rLrp3bU13HcCF6G20RcLfc8pds+7bNyVB9OP/9bO+endldd95RSCfEmQCTLBMA4+HLk8eyzZvWF9J4/vn1yQbmvTe3kEb0HhDcW7dszL7+6kT20ksvFuIBwbV2zars+LHD2dw5swvxrXLTjf2yQwf3ZUu/WFyIE+2FMXjs6OFCuBC9jbYIOARbvxuuz6+ZgBgEMZ0QfZmzzjor+2zRwgbBdvbZP8qWL/s8m/LC84X0cO65v8hGjri/EC56Nw88MDLbt3dXEugxDj5f8lmyixLjpxeE9IoVSwvh7aQnniFEGV0WcBdf/Ots1arlafKyMLaAjh09lF155RWF9EL0VW6++c/ZwQN7s/fnvdsQ/tRTT2br1q5OYi7ew8Ln2mv/WAgXvRsE3Ltz30leuLJ23bN7RzbnnbcbBBx29cUXp2SHD+1PTJ0yOdnNC84/P9lQPGzA9RNPPJb6EtdvzHgtGzCgf7bo00+ySc89k/IaP25sWijcc8+QbOHCBcnTh6B86KEHsunTp6X88SLNmvlGSo/IYGtwpRMaV1/9++zokYN5nnZNmR97dHx6NwQq28T9+9+S/iaMHZf4vuRPPs3yt3d4ftKzDflfftllDXnxPN6dOWTN6pXp2RZHnfGeJ08cbYiz99u2dVP23rtzsq++PJ7qZtPGdSmt1S1paQfENXXEv+Tp38Py8e9AnZAX5SXO6iA+g7FOe9jzbGy//tor6Zr38u8qRFfosoCjw5atMvE62MAV4kxg5pszkuc59vsLL7wgTbBM+vEePHO2dbpi+RcNk5XovdCWbGu+9dbM5G3zC9jbbh2U3Td8WOoHZhtZ6DLxm4gAhAKTv12zLWtbd6TfuGFtwzNZGFjfuuSS32Tbt2/JBg78ax5PORASdo1QQiTZNYIleopif+V6/gfz8vfhPTasX5uf9aNcvAPv7vMBRE1H+fMOCFubY1j0UC/kyzMXLJif6sXSc6aascPfiN/hw4bmcY+MH9fwvrzfieNHUv1bGOX0ZbJ2sPLjZOB5eMJ9PibggHegbu3a2tHyiM+wPHbu2NYQRpoysS/EqdJlAeeNlIewOJEJ0ZexowNl/b5KwDGh4GHBQ8e9fkISvRcTcHhQ8RR5YUE/YKKOtpHzcnhn7Boh4IXCww8/mAQcogIvEvhnepsahYjFm9ixMvprREcUGrG/xjOYZeKkSsCVpY35x3nB3oN/EV4IH+/B5kORjz78IHnpEEReANm5QBZIVq7du3Z0WCba4YrfXV54vsWTTxRwvk7IryMBh+DDC2jXlHvixAkNaYToKm0RcKySYngcqEL0dUzA4YmLcVUCLt7PFs3IkToT19sxAcfftDkig78RdOZBigLOgyA5cvhAg1CAV15+KX0Qhmcubi32dQGHWMNjWDZvcE70+LEj+dYkUH+kN/EVhVdVmTxsN5NPuwXc2DGjkzfQPJmIUx0pEu2mywLujttvS+cI4jkeVlHExfRC9FVsS8dviwFGnEm9MwacM0xxMhC9Dy/g8LYwybOdivgyT4sXcPQB4hFUtl0XPXCAFw5xsH/f7vQzJD6urws4O4JQtgAycRfDPVF4QSyTtQPPoB26ywMHHI/gWWw/c8YxxgvRVbos4HAN4yr27mHCqg5tC9FXsW2e+BMECDfz0BiMsbLfS2TSLZsMRO/CCzjOidHmLGS3bN6QiwEv4Gx70E/kUcAh2DjXxrliPLH+gwDo6wLOFkBlP7tjH8ZFr6QnCi+IZbJ2MM9Ydwo4swe817KlSwrxQnSVLgs4sK9+bELiMO6CjxonLCHOBBgDGGu+OmOS4PcQmdgtnm0gJgTi+H0wPHM23vAyIASaTVKid8BXovxIs13bltn1f7ouXdO+s2fPytavW5P6hAkH2pdrfiuQ9HZmizBLyzVbsXxtiUfOnsGuhv0cjX3E4L8I5bcH/RlKyuivEUYIQ8TIuLFj0vPoizt2bM0FDdf+DBq7KL5c/MuWY9nuCp6mjvL37wAsbqiDa675Q7q2c4ArVy7Lpk2bmn5Lz8YPAhcv3OpVK7IhQwZns2a9mb7qtLypX+rEl4l6RGyNGjUipbN2YKzxLgg12sGeX5ZPrBOcE3yZOnjwXYVn+GdbXrSj/5ktIdpFWwQc6b/dHvj2jAJ/t5qHEH0Fm8jwGDAennt2Yh6HMTevCBOKbQ0xIRDuD7mL3gmCxH4Cg78Jo82954j2JA2TP/2AMBPohPEzIwh40sydOztdkx+eHEQgwt6egTBDaNg1vzVI/uSDzcVjx89d2POI53mk5Zp7iUc0Hti/J4VxP191cg99EgHDs4kjDOH06ivT8wUH9/3tvuH5/YTHeoGq/G+/7dbCOzz6yLiG97A8hg27N38XROCN/W7I4yjXv+r1cDrDRjhlJQyoD18mRCzhJmZpB0vLeVPawerR5/P0hKca6oSPJe6+e0j+fC+O7Rl8kOSfzfZ69L4L0S7aIuCEEEII0Qjewmb/3ZoQXUECTgghhGgjeNeffPLxgjdQiHYiASeEEEK0CbZZbZs4xgnRTiTghBBCCCFqhgScEEIIIUTNkIATQgghhKgZEnBCCCGEEDVDAk4IIYQQomZIwAkhhBBC1AwJOCGEEEKImiEBJ4QQQghRMyoFnBBCCCGE6J1UCrgYJoQQQgghegcScEIIIYQQNUMCTgghhBCiZkjACSGEEELUDAk4IYQQQoiaIQEnhBBCCFEzJOCEEEIIIWqGBJwQQgghRM2QgBNCCCGEqBkScD3ERRf9Kntk/LhCeN0Ycf/fC2FCCCGE6Fkk4E6Bs846K7vyyivya+rLX5exaeO67KmnniyE143r/3Rd9uCDowrhQoj2g2255JLfFMKrOPfcXxTCRDXY8muu+UN2xe8uL8SV0Upb1A36zsUX/7oQLnovXRZwQ4fek3391Ynsn1+fzCY990z2y/POy+NmvP5qdujgvuzjjz9M6eK9deTY0cPpXXkvrl966cV0vW3rpkJa49FHxmXDhw0thJ9u3nt3TqHcPznnnGzhwgXZy9OnZW/MeC07eeJo9uKLUxrSbNywNpv33txCfqJ1mED27N6RzXzj9Wzrlo1pLDUbf9u3bc6WL/s8+3D++9nxY4ezu+68o5BGdC8vTJ6UtxXjP8YD42jtmlWpjebOmV2I7wxmW1asWFqIizD5mh2OcTfd2C/Zq6VfLC7EnanQPsuWLsmuuurKdN3vhuuzfXt3fdNeRwpp4c47bm+w+90NcynPmzhxQiEuponhreL7Tk/Y9ao5BlsY04rmdFnAwepVKyo7UmeMT53A07Zjx9aGgXz40P6CEDIQtAie3tQ5R49+KIk3Bm0sN+FbNm/Ir5movjx5rCENRoV3jvmK1pk6ZXI2aOCA9LdNKrNmvlFIB5dfdllqH+tLDz/8YLZ3z87kFY1pRffywAMjk83bvGl9IQ7GjhmdnTh+JFuyeFEhrhUYZ521oUyCZXaYfsUietTIEYW4MxV2Q7768nh+feGFF6R5jAVSTGsQ304BN37c2NQuMRxuueUvyQ4z5n16n+ahhx5IC4R476lgfacnBFzVHNMXdqh6mrYIODrZzh3bsvuGD2sIZwW5atXy/BpDwgBglfPZooXZpZf+NoVfffXvs6NHDqbOw4rowP493xib+1NaII8Lzj8/W7BgfrrGYzFs2L0Nz+KawUc+x44eyvr3vyWFs0rZtXN7Eiorln+ROgrpbh00MHUkPGoYyWcmPt2QH1uehPM8VgsWjouZvPxA5u8ohAwG6MEDewvhb701M92HkacuYrzB2TneCUg3a9abqcNjcGLaVqHMsdy8M8bDrlmdURf+/B4CYuU3k0pH28ZnIjf2uyH7fMlnqW2pIwv/233D0yrep31+0rOlK34MKatTH2b9Li4E6Ov075iH6F4QcO/OfSctgs4++0eFeGzUnHfeLkyI2Nb9+3andmdcRTt27bV/THH0ATwTTNBewGFDsR2kwXaYDYUqj8yAAf2zRZ9+kuIt7MknH0/in/7DmPf5AOUy+4uXhDCze2aPrSxldoznmYeY59x666B0n7fppPM2PZYbqCMr43Q3Jigf4pl6wHPmdwmw69hL0nC/LVTNrlFu5hjCrTzAPd5Wk+71115J6YD38HYfWIDxfNpqzeqVebjNOzff/Oc0f8R55p57huRtTJ5RvCDWeD+8p1zzHuYBtPTEWb1hFyjfkcMH0nMWfPRBQ34G8yIefOqNMtmOmfUd31+r5kCgjnkWbcx7mmaI/aqsHFVzDHUR04rmtEXAAQP5ow8bGwvPk7mAEXlsKZihwIghRGg8XPt0Hu7nHjoMnZMJkHA/aTFQEV/x+Qw+G+B0buuIdDDywMDM/2BeNm3a1DQY6XQYRq5Jw/2W18hvxCPPZgDjHaGsZqRbEXCWls7pw6kLDBh1gYEjf7/S8umoIxtkDDjeg3vbcdYlCjjy3L59S+GcB3UZt4FmvjkjGzni/kKeZzKI6g3r1yavy7ixY1L/wlitW7s6GbUo4EhX5smkf2JkfRhHEDCKMS3tVyYCRfeCgKNNWLjyt4/DXiFMEOh+QsSeYOvMY8oE7SctPGQICBa0lp4Fpxdw2FATP9gO7IPZjioBh7eDPuUFHH2JRRllRaT4OAOR48vHvIAXinL5skQ7xnky25Kj3/MvtpD7sKvYYW/TeU6ZTZ/ywvPZ008/ldJyD3laHOPGyk8d400jf78dyLPfn/duKjNh69etSffyrN27dqR78HgD5aJOvF3n/RBnPANMPFk8z8MZgCjC8UBbmDfd5h3uJ/84zyCKseW0N/HRO0o67jcBRz2T3spLeuy09QX6mi2yac+4a2JQbzyb92G+RHTzdxRwzeZA/mWe4J1xMKxcuSwvZ+xXsRzN5piyviua0zYBR4MxSOhIFoao49/bvll90XG8uxij592/NB4q3w9shAtGzjx7GAcEXnw2MBhtpQh25s7Of3ihgkH0AzEaPu71A4rOZUa6FQFn7+iNuNVFTBcnbKgqZ1zVe/D02EoxgiHznrso4KgrruNBVsoXt3EIa1YO0ZyyfmSU9afYRw0ZvtMD/Z/xwtkpJi1/xpU2YZKjzWyM2C4FYt7nw6KVxRB/M9l5+wn0BRt72I645YZ9MdtR1UeAcphIwwZw34QJ/8jj8RTFe7C73kvDxGwiqKws3o5V9UvyIk9v0+PODVAv3k4iCs0LRNwdt9/WkJ4taxbKCISyeoi2lPL5+SeOR9oVAePnI5+Htaf3vtIfbHfE5h1vS2O5fNtGrDwmjCy/mA54Hs/17Vl25pwdMbZd7Zq6sjOccW5pNgeycOfd2W3gmjqgXcv6VSxHszmmXdvBZxJtE3DACgGRwN8MAAY5f+M2RtzZCgdYYfgVVZUwoYNZOIO0qsNj/MgDTxHPtvBTEXCeMWMeTivYdgk4qwurB6uLsvcizHtdKGcUyRFWRZOffy6bOvWFAk88/miDwSkTcIg/tnF8nhJw7cf6UZlXraw/VfXRqolSdC8m4Pib+rfdB2yPbWl7Acekh6c0ih52C0z00BfihOcneWyHt6GAPbD4qj4CXsCBbQu+8vJLpd5/wFZgz7G7XOMxtgm/rCzejjXrlzzX2/SyLWgESRwDYGIlHt/AZuHJo/7L6iHa/I4EHO1iwrosD2tP//5sKZo46UkBB5ybpV7ZUq1qT/qZ5ReJAs4T50DEGmmZcxH15pGFjvpVszlGAq512irgGNw0Hg3sV5qIKho8DghPVecBvGtsxZKnF2dl2PkHPHWU41QEHCsPXNP2LL/6iAMdYv4G92NU/CC1uohpy2Bg4OLGcCLA6PjR1d4VooDDkFLX8TwGq7Z4Jos6i+cGz3Ra3UJlMi/rC9FbDXgcmCCi4WOLiPCYh+hevIDjzCPtyI4DW2p2bMQLODs0Hyds8qG/YFdox+hZ8pM8tqOZDY12zBMFHOBdsrNqtpUW72MSZhsQW8aXzxbeUVmaCTjiqAvyjB5Jg/ECMZz+T3gUItSfCYOyeog2vyMBR3y0eT4PW4j7eE9PCzig/dgetu1b3552ziwuEIwo4JrNgQbzEztk1IOfM3y/iuVoNsdIwLVOWwWcfYrN2QU/MM14mXeujGYCjgmNe+MZO48/Q4EXis5Op29VwNmq0x+yPFUBZ4OGFaOFWV2UrTojfLDAOQy2hpnQ/UqnHUQBBwxGJiQfhiEzb6qBAa8yBmcqtGn0DLCdgPguazv7UtH/9A7g2TCvh8+bfhl/VoBx1mxcie7BCzjbakRA4KmwSdsLODvDGD84QQiYR90Wqj7eT/LYDtq6ynZEgeDxAg675H/3bPDgu3LPUbwPsAkszv1WZ0dlaSbg7BjJc89OrLTp9Gv/lahR9lEV+C3PsnqINr8jAUe7YAertlCtPcu8TNCTAo468ceHaE8cB7E9mUPsWFPEC7iO5kA/19pZOuLL+lVZOarmmHi0SHRMWwUcMCARcf73key8QDzQSAcxA9BMwNHYGMg4qXlixzR3eqsCzgaO93Tg5TgVAQeUy3tIrC74esmn4zfzfvaz/5c99uj4/GwBB1zjqtlg4Pi0p0KZgLMvmezaziHGVRQGNgoP0Rr2EzP8TqCFUc/mPeZwOH3P6p5JE0HvxSBtFVezovvxAs4mPBM6lsYLOEvjbSDtim2zjxq4n/b1z2FSs0ke28H93nZgQ7Ed5B8FgscLOGyYF4pcN/uynTKAF2tlZQErSzMBZ+/NzzFV2XTbzbHfaQP+Jm/ioseQ97FxUFYP0eZ3JOAQmYwt/zuL9hUtf1t7+p/1ARbd5NWTAo60/mc57N7YntQ18yLCysIQfqT3Aq6jOZB//U4Y1yzoy/pVWTmq5phmv8Ygymm7gKNjV62q+KybQUlHoSPxyTGqnU5MGHEM6ngfDRy/5IywsuVMAp9yszLi6xjC7ZN84ID/3XcPycvAuQE6l11TDsqDUbR73nzj9fQ/D2Cs6GD2dRBgMF59ZXpD/rFcDG7ui2KH9/R1QTobiKxoSMO/eGgsfwMxG9O2AuXGIFt+UQTgzuZ9zfPpjSggSsu2N8SpwbYb3gY7Q2PheGZoB/sqEb79CYnDyZjTF7syVsWpYWOXtjF7ZZOQpbEzvt+O13+dc8R7wZedCCbu9T8jgs1iO9XGJTaNdma8X3fdtSkN6bnP0mBDsR3YLTsHjAiMHytZeU34YHMoI1v+lK/MQ2ww8XuBYFhZoh2jv1oY9jhO4EB9Vf0Isk9j23DUgf+pED5ooB5xFmCnzA6W1QO2ytt8BHO0+dGukxfPsx9u518EHHnzO5pWDp7xr3Y+nI9Hm3eoE+aFsnKxBc07gPeggZWHfKw8pCcspqfOEdJ2BpH2jL8aYFBPljf52Je7VjbehbI1mwPpp5SL83Zgc0FZv6oqR0dzjOgcbRdwQIeIYYatTmJ4M1jtxZVpGTyXlUEzY9RZ+O9V/A+kdiVfjLb3shjURfwvXHzd87k1g9a2ghlUrJDo+AjlrrZTMyhXVTtRrvi/M4iuQV+AGM7vbcUw2p2+2Z3tL7qPjmwgcWYX2JL3Hp6yNKcK5ehMPyI+/uxDzKfVsmDT47GMMpr1dewxdrksrl3451PmsjmAeYf5oqydOoJ7y8Z9FbR7VXrrVx3VB+XszHzWbA4kj6r/gqyz/arZHCM6R7cIuHZCWfgNmt74X1G1Aq52iOFVMAhY8UWjgPvb/2ZdT8Ov/1ed+xBCiI7AC4RNj+FCiNbo1QIO9yrepnjOoI5Qp6xYYngz2D7AfW3X9//9b+kHKP0v/Pc0ZSsuIYToCDu6wTZdK4tZIUQ5vVrACSGEEEKIIhJwQgghhBA1QwJOCCGEEKJmlAq473//+4UwIYQQQgjROygVcN/97ncLYUIIIYQQondQKuDgP//zPwthQgghhBDi9FMp4L73ve8VwoQQQgghxOmnUsDBd77zHXnihBBCCCF6GU0FHPBBA2fi/uM//qMQJ4QQQgghep4OBZwQQgghhOhdSMAJIYQQQtQMCTghhBBCiJohASeEEEIIUTMk4IQQQgghaoYEnBBCCCFEzZCAE0IIIYSoGRJwQgghhBA14/8DXDsEAV3dWy4AAAAASUVORK5CYII=>