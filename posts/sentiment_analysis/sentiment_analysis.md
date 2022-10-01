---
title: "Sentiment Analysis - Rotten tomatoes reviews"
date: 2022-10-01T17:46:41+03:00
draft: False
author: "Yuval"

resources:
- name: "featured-image"
  src: "danie-franco-Zi8-E3qJ_RM-unsplash.jpg"

tags: ["Python", "NLP", "Bert"]
categories: ["NLP"]
---
This project is all about sentiment analysis as the title suggests. We are going to do so by using a state of the art model, called BERT (developed by Google).

<!--more-->
![Tomatos](https://images.unsplash.com/photo-1635848831260-97b70d5bed58?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1170&q=80)

We'll start by installing amazing library, primarily applied in Natural Language Processing (NLP).
```Python
  ! pip install transformers
```

Importing some other packages and libraries:
```python
  import pandas as pd
  import numpy as np
  import requests
  from bs4 import BeautifulSoup
  import re
  from transformers import AutoTokenizer, AutoModelForSequenceClassification
  import torch
```

For our task we'll use **bert-base-multilingual-uncased-sentiment**.
{{< admonition type=quote title="As stated in the official HuggingFace website:" open=true >}}
This a bert-base-multilingual-uncased model finetuned for sentiment analysis on product reviews in six languages: English, Dutch, German, French, Spanish and Italian. It predicts the sentiment of the review as a number of stars (between 1 and 5).
{{< /admonition >}}
{{< link href="https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment" >}}

```Python
  model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
  tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
```
All our tokenizer will do is to convert a sentence to a sequence of numbers.

The function bellow is our main function in this code.
```python

def sentiment_to_number(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1
```

#### How it works:
We will understand how it works with the help of an example:
{{< image src="/posts/token.png" src_s="/posts/token.png" src_l="/posts/token.png" >}}
{{< admonition type=example title="Step 1:" open=true >}}
The tokenizer.encode returns a torch.Tensor object containing numbers that represents the given string.
{{< /admonition >}}

{{< image src="/posts/model_token.png" src_s="/posts/model_token.png" src_l="/posts/model_token.png" >}}
{{< admonition type=example title="Step 2:" open=true >}}
Calling the model with the token from *Step 1* will result in a SequenceClassifierOutput type object. The important part here is the tensor containing numbers which represents the probability of a particular class being the sentiment that corresponds the most to the review given.
{{< /admonition >}}

{{< admonition type=example title="Step 3:" open=true >}}
torch.argmax will return the index of the biggest probability [0,4], than adding a 1 to it will give us a score between 1 to 5.
{{< /admonition >}}

#### Scraping the data from 'Rotten Tomatos':
In this example I used comments given to the new "The Lion King" movie.
```Python
r = requests.get('https://www.rottentomatoes.com/m/the_lion_king_2019/reviews')
soup = BeautifulSoup(r.text, 'html.parser')
regex = re.compile('the_review')
results = soup.find_all('div', {'class':regex})
reviews = [result.text.strip() for result in results]
```

#### Creating pandas dataframe:
```Python
df = pd.DataFrame(np.array(reviews), columns=['review'])
```

#### Adding a score column for each review:
```Python
df['sentiment'] = df['review'].apply(lambda x: sentiment_score(x))
```
{{< image src="/posts/reviews.png" src_s="/posts/reviews.png" src_l="/posts/reviews.png" >}}

Another small trick we can use, adding background-color for the 'sentiment' column:
```Python
df.style.background_gradient(cmap='RdYlGn').set_properties(subset=['review'], **{'width': '300'})
```
{{< image src="/posts/colored_sentiment.png" src_s="/posts/colored_sentiment.png" src_l="/posts/colored_sentiment.png" >}}
