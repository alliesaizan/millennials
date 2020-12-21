# The Millennial Question (Data Work) :thinking:

## Overview
This repository contains the programming script that cleaned the data behind the Pudding article named "The Millennial Question", published in October 2019.The finished article can be found here: https://pudding.cool/2019/09/millennials/. 

## About the Data
I used the [Event Registry News API](https://newsapi.ai/) to extract news articles with "millennial(s)" in the article title. I focused the API call on articles published from 2015-2019. A sample of articles extracted in the API call can be found in the file *Sample Articles.csv*.

## Data Cleaning Process
All data processing takes place in the *article_processing_script.py* file

## Part-of-speech tagging
The goal of the data collection was to identify the actions Millennials were taking. So if an article reported about how Millennials were not buying napkins, I needed to identify (1) that Millennials were the subject of the sentence, (2) that the action they are taking is "not buying", and (3) that the object of that action is "napkins". I used the scrapy package to identify the following parts-of-speech in each article headline:
1. The subject of the sentence. I only kept articles where Millennials were the subject of the sentence.
2. The subject verb. For all sentences with Millennial subjects, I extracted the verb.
3. The direct object(s) of the verb. For all verbs, I extracted the associated direct objects.

We focused on article headlines because they are often written to evoke strong emotional responses about the norms that Millennials are purported to be breaking.

I exported the results, such as samples of headlines, verbs, and objects, throughout the process for intermediate analysis. I  completed minimal text cleaning, including removing special characters and snowball stemming the verbs (to remove duplication). Finally, I used NLTK's SentimentIntensityAnalyzer to obtain headline polarity. In other words, I determined whether the headline had a positive or negative sentiment. 

Sample noun output for the verb "kill" is below:

 *kill: {'team loyalty', 'fresh perspective', 'its turnaround', ""the 'pretty robust' housing market"", 'prenups', 'scion', 'bruce wayne', 'industries', 'a whole  host', 'our shopping habits', 'mayonnaise', 'the jewelry business', 'the boozy 18-30 holiday', 'the purpose', 'applebee', 'napkins', 'it', 'bar soap',      'divorce', 'the retail store', 'diet coke', 'the wine cork', 'actual gangs'}*

## Transforming the data into a consumable format for JavsScript
The data cleaning process outlined above yielded a data frame with one verb and a row for each noun (sentence object) associated with that verb. Data frames are not an ideal data format for web development, so to get the data to the next stage of the process, I needed to transform it into a JavaScript-agreeable format. Through  series of loops, created a nested dictionary that contained multiple levels: each verb key mapped to a dictionary of objects, and each object mapped to a dictionary of articles containing that verb-object combination. I included the article polarity at the most granular level. I removed dictionary entries with missing articles and exported the nested data to JSON format. On the website, this nested format enables users to expand and collapse information based on each verb.

## Final thoughts
I am really happy with how this article turned out! I'd had a goal of freelancing for The Pudding for a while, and helping to create such a cool reader experience is this Millennial's proudest moment of 2019. 🎉
