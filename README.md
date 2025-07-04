### AMAZON REVIEW CLASSIFICATION : A Machine Learning Approach

#### Abstract
This is a final project for my Machine Learning and Statistical Data Analysis course (CS74). I analyzed the Amazon product review dataset and create classifiction and clustering approaches models. I used skills and concepts learned during course, such as feature engineering, and model evaluation metrics. The project also required to produce csv files from each training which was used in Kaggle to automatically compare my results to a hidden "correct" test file.

The objective is to accurately apply machine learning models and concepts learned in this course to a real-world dataset. The Amazon reviews dataset was analyzed to predict product ratings from textual reviews and analyze performance using multiple evaluation metrics. The project involved several stages, including data preprocessing, feature engineering, model building, using Scikit-learn, I trained binary and multiclass models, tuning hyperparameters via cross-validation. Results are benchmarked against baseline F1 scores using confusion matrices, ROC curves, AUC scores, and macro F1 metrics, and clustering methods to analyze the product reviews.

#### 1.  Introduction
The task of review classification is central to understanding user sentiment and product quality on e-commerce platforms. This project explores binary and multiclass classification
models to classify Amazon product reviews using natural language processing techniques and classical machine learning models.

#### Dataset
The dataset provided for this project are Training.csv, and Test.csv. The training data contains 29189 rows and the test data contains 4500 rows. The test dataset includes the same variables, with the exception of the "overall" variable. The most useful data from the features includes: "reviewText", "summary", and "category".

#### 2.  Dataset Description
The dataset comprises Amazon reviews with the following fields:
* overall: The rating given by the reviewer (1 to 5 stars)
* verified: Boolean indicating if the review has been verified by Amazon
* reviewTime: Date of the review
* reviewerID: Unique ID of the reviewer
* asin: product ID (one product has many different reviews)
* reviewerName: Reviewer’s display name
* reviewText: Full text of the review
* unixReviewTime: Unix timestamp of the review
* vote: Number of people who found the review helpful
* image: Link(s) to images included in the review (if any)
* style: Dictionary containing product style info (e.g., color of phone, size of shirt)
* category: The Amazon product category (e.g., Electronics, Clothing)

##### Preprocessing
I cleaned the data by removing missing reviews and normalized text using Scikit-learn’s TfidfVectorizer with a mindf=5 to exclude rare tokens, max_df=0.8 to exclude overly common tokens and a max vocabulary size of 10, 000.

## 3. Binary classification

#### 3.1 Labeling Strategy
Binary labels were created using four cutoffs:
* Cutoff 1: 0 if ≤ 1, 1 otherwise
* Cutoff 2: 0 if ≤ 2, 1 otherwise
* Cutoff 3: 0 if ≤ 3, 1 otherwise
* Cutoff 4: 0 if ≤ 4, 1 otherwise

#### 3.2 Models Used
I tested the following classifiers:
* Logistic Regression
* Support Vector Machine (SVM)
* Perceptron (cost function)
* Random Forest
* Naive Bayes

#### 3.3 Hyperparameter Tuning
Grid search with 5-fold cross-validation was used to tune parameters. Each model’s performance was evaluated on a held-out validation set.

#### 3.4 Evaluation Metrics
For each cutoff, we report:
* Confusion Matrix
* Accuracy
* Macro F1 Score
* ROC Curve and AUC

## 4. Multiclass classification
#### 4.1 Setup
I extended the task to predict the exact rating (1–5). Models were trained using one-vs-rest
strategy with label binarization for ROC analysis.

#### 4.2 Evaluation Metrics
* Confusion Matrix
* Accuracy
* Macro F1 Score
* ROC and AUC (One-vs-Rest)

## 5. Clustering
To cluster the product reviews in the test dataset, I created word features from the data and used that for k-means clustering. Clustering will be done by product types, i.e., in this case, the labels will be product categories. Silhouette score and Rand index were used to analyze the quality of clustering. Silhouette score of 0.7947 suggests the clustering is performing very well in terms of structure and separation. Rand index of 0.5371 means the clustering agrees with the ground truth only slightly better than chance.

## 6. Discussion
Logistic Regression appeared as the best model for binary classification. Logistic regression is often considered a strong baseline classifier, not necessarily "the best overall," but consistently reliable, interpretable, and efficient. While Naive Bayes (multinomial) appeared best for multiclass classfication. Naive Bayes (especially Multinomial Naive Bayes) is often considered a strong baseline for multiclass classification, particularly in certain domains like text classification (e.g., spam detection, sentiment analysis). While it may not always be the best in every scenario, here’s why it often outperforms or competes well with models like Logistic Regression or Random Forests in multiclass problems.

