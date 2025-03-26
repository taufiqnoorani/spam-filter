# Spam Filter

Spam detection is a classification problem, as you have to decide whether a message is spam or ham (Ham is a word used throughout literature meaning 'not spam').  

Using Bayes' Theorem, we'll implement an algorithm called Naive Bayes.

## Table of Contents
- [Bayes Theorem](#bayes-theorem)
- [Naive Bayes](#naive-bayes)
- [Using Naive Bayes for Spam Filtering](#using-naive-bayes-for-spam-filtering)
- [Arithmetic Underflow and Mathematical Flaw](#arithmetic-underflow-and-mathematical-flaw)
- [Implementation](#implementation)
- [Dependencies](#dependencies)
- [License](#license)





## [Bayes Theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem)

Bayes Theorem is a way of finding a probability when we know certain other probabilities.  
(Learn or refresh the [Basics of Probability here.](https://youtube.com/playlist?list=PLvxOuBpazmsOGOursPoofaHyz_1NpxbhA&si=ecHB8oh4FReS_TXv))   

Bayes theorem is stated mathematically as the following equation:

$$ P(A \mid B) = \frac{P(A) P(B \mid A)}{P(B)} $$

### Example: [Picnic Day](https://www.mathsisfun.com/data/bayes-theorem.html)

You are planning a picnic today, but the morning is cloudy.  
- 50% of all rainy days start off as cloudy.  
- But cloudy mornings are common (about 40% of days start cloudy).  
- And this is usually a dry month (only 3 of 30 days tend to be rainy, or 10%).  

The chance of Rain given Cloud is written as:  

$$ P(Rain \mid Cloud) $$

So let's put that in the formula:

$$ P(Rain \mid Cloud) = \frac{P(Rain) P(Cloud \mid Rain)}{P(Cloud)} $$  

Substituting values:

$$ P(Rain \mid Cloud) = \frac{0.1 \times 0.5}{0.4} = 0.125 $$

That's a 12.5% chance of rain. Not too bad, let's have a picnic!





## [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)

The Naive Bayes classifier is a popular supervised machine learning algorithm used for classification tasks, such as text classification. Unlike discriminative classifiers such as logistic regression, it doesn’t learn which features are most crucial for distinguishing between classes.  

A Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature, earning it the name 'Naive' Bayes.

Despite its simple design, Naive Bayes has demonstrated strong performance in many complex real-world scenarios. A 2004 study on Bayesian classification provided theoretical explanations for its surprising effectiveness. However, a broader comparison in 2006 found that Naive Bayes was outperformed by more advanced classification methods, such as boosted trees and random forests.

One key advantage of Naive Bayes is that it requires only a small amount of training data to estimate the parameters needed for classification.





## Using Naive Bayes for Spam Filtering

Since a message is comprised of different words, we can calculate the probability of a message being classified as spam given that it contains a certain "trigger" word. Computing the overall probability for the whole message is then simply the multiplication of all those individual probabilities.





## [Arithmetic Underflow and Mathematical Flaw](https://muens.io/naive-bayes/)

### Arithmetic Underflow
Multiplying small probabilities can lead to "arithmetic underflow," which turns numbers into 0 due to precision limitations. To handle this, we use logarithms:

$$ \log(ab) = \log(a) + \log(b) $$  
$$ \exp(\log(x)) = x $$  

### Mathematical Flaw
If a word has never appeared in spam messages, then its probability becomes 0, making calculations impossible. To resolve this, we introduce a smoothing factor `k` (typically set to 1).





## Implementation

### 1. Data Preparation
The dataset used for training the Naive Bayes classifier is [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset). It contains a CSV file (`spam.csv`) with labeled messages categorized as either `spam` or        `ham` (not spam). The dataset is loaded using Pandas and preprocessed as follows:

- The `label` column is renamed to `is_spam`, and values are mapped to `True` for spam and `False` for ham.  
  (Originally, the columns were named `v1` and `v2`, which were manually changed to `label` and `message`)
    
  ```python
  csv_path = "spam.csv"
  dataFrame = pd.read_csv("csv_path", encoding = "latin1",)["label", "messages"]
  dataFrame.rename(columns = {"label": is_spam, message: "text"}, inplace = True)
  ```
  or
  
  ```python
  csv_path = "spam.csv"
  dataFrame = pd.read_csv("csv_path", encoding = "latin1",)["v1", "v2"]
  dataFrame.rename(columns = {"v1": is_spam, v2: "text"}, inplace = True)
  ```
  
- The `message` column is retained as the text data.
- The dataset is converted into a list of `Message` named tuples for easy manipulation.  


### 2. Tokenization
- To process the text data, a `tokenize` function is implemented. It extracts words from the message, converts them to lowercase, and removes short words (less      than 2 characters).  


### 3. Splitting Data into Training and Testing Sets
- The dataset is split into training and testing sets using an 80-20 split.  


### 4. Naive Bayes Classifier
-  The `NaiveBayes` class is implemented with:  

- **Training method:** Counts occurrences of words in spam and ham messages.  
- **Probability estimation:** Computes word probabilities using Laplace smoothing.  
- **Prediction:** Uses logarithms to avoid arithmetic underflow when computing spam likelihood.  
  

### 5. Probability Calculation
- To estimate the probability of a word appearing in spam and ham messages, Laplace smoothing is applied.  

    ```python
    def _p_word_spam(self, word: str) -> float:
        return (self._k + self._num_word_in_spam[word]) / ((2 * self._k) + self._num_spam_messages)

    def _p_word_ham(self, word: str) -> float:
        return (self._k + self._num_word_in_ham[word]) / ((2 * self._k) + self._num_ham_messages)
    ```


### 6. Prediction
- Given a new message, the classifier calculates the probability that it belongs to the spam category using the log probability trick.


### 7. Testing the Classifier
- A test function validates that the classifier correctly counts spam and ham messages, processes words correctly, and predicts probabilities accurately.  


### 8. Running the Model on Real Data
- After training, the classifier is used to predict whether new messages are spam or ham.
  
  <img width="1525" alt="spam" src="https://github.com/user-attachments/assets/cc339aec-8374-45cc-8d53-a95587c92f6c" />  
- 99% sounds too good to be true and there's certainly a smell of overfitting in the air. The data set we've used for training is small.
  
  <img width="1525" alt="ham" src="https://github.com/user-attachments/assets/d8c406ca-6a78-4ed1-8b61-348c4ebf496d" />  
- You might want to modify the code to train on the larger messages or find a different data set altogether.

This implementation ensures the classifier is trained efficiently and performs well on predicting spam messages using Naive Bayes principles.


## Dependencies
Ensure you have the following Python libraries installed:

```bash
pip install numpy pandas scikit-learn
```

- **numpy** – For numerical operations  
- **pandas** – For handling datasets (loading and processing CSV files)  
- **scikit-learn** – For train-test data splitting  

### Prepare the Dataset
- Ensure `spam.csv` is in the same directory as your script.
- The CSV should contain at least two columns: `v1` (renamed to `label`) and `v2` (renamed to `message`).



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
