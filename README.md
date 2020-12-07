![](https://img.shields.io/github/repo-size/kiranrawat/Detecting-Fake-News-On-Social-Media)

## Fake News Detection on Social Media:

### Overview:
As we know how fake news on social media about Hilary Clinton had a big impact on election results.  The cost of publishing information in public is very low and the efficiency of spreading news through social media is very high. 

These benefits can also enable spread of low-quality news with false or fake information. 
During 2016 USA presidential election, one of the most escalating news was the one that claimed Hillary Clinton ordered the murder of an FBI agent and was viral on social media.

### Motivation and Background:

- The extensive spread of fake information on social media.
- When social media has become the most cost-efficient way of communication among people, it is extremely intriguing to analyze people’s reactions to a popular news post while eliminating false information online. Therefore, designing a news monitor system that concentrates on the news content to alert the public about fake news.

### Goals:

- Guiding people on their thinking over false information.
- Identifying fake news over social media.
- Building a classifier to predict news as a Real or Fake.

### Datasets I intend to use:

https://arxiv.org/abs/1705.00648 [cs.CL]

### Data Science Pipeline:

- Data Collection : Balanced dataset collected from politifact.
- Data Preprocess: Data Clean and Natural Language Process
- EDA and Feature Selection : Binary, CountVectorizer, TFIDF
- Model Selection : Naive Bayes, Logistics Regression, SVM, RF
- Model Training  : Scikit-Learn
- Inference : F1-Score and Confusion matrix to make an inference
- Model Deployment : Deployment on AWS or heroku
- Data Product : Flask-based web application

### Some Practical Applications:

- Social Media Websites (alerting Fake news)
- To protect the nation's ecomonomy (For example, fake news claiming that Barack Obama, the 44th President of the United States, was injured in an explosion wiped out $130 billion in stock value [Rapoza 2017]. ) 
- Survey of fake news

### Libaries to install

- `pip install -r requirements.txt`

### Train the best model

- `python training.py`

### Run the Flask Application

- Start flask web server: `python app.py`
- The server will start on the address http://127.0.0.1:5000 [if port 5000 is not occupied]

### References
- K Rapoza. 2017. Can âĂŸfake newsâĂŹ impact the stock market? (2017).
- https://kavita-ganesan.com/news-classifier-with-logistic-regression-in-python/#.X7XeFBNKhQK

