### AMAZON REVIEW CLASSIFICATION : A Machine Learning Approach

This project applies various machine learning models to the Amazon product review dataset to perform binary classification, multiclass classification, and 
clustering approaches to analyze and categorize product reviews. The objective is to accurately predict product ratings from textual reviews and analyze 
performance using multiple evaluation metrics. Using Scikit-learn, I trained logistic regression, SVM, and perceptron models, tuning hyperparameters via 
cross-validation. Results are benchmarked against baseline F1 scores using confusion matrices, ROC curves, AUC scores, and macro F1 metrics.

#### 1.  Introduction
The task of review classification is central to understanding user sentiment and product
quality on e-commerce platforms. This project explores binary and multiclass classification
models to classify Amazon product reviews using natural language processing techniques
and classical machine learning models.

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
I cleaned the data by removing missing reviews and normalized text using Scikit-learn’s TfidfVectorizer with a mindf=5 to exclude rare tokens, max_df=0.8 to exclude overlycommon tokens and a max vocabularysize of 10, 000.
