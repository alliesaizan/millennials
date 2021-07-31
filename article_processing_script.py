#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 14:46:48 2019

@author: alliesaizan
"""

##############################################
# Package import
#from eventregistry import *
import spacy
import requests
import bs4
from nltk import sent_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import os
import re
import pandas as pd
pd.set_option('display.max_columns', 500)
import pickle
from itertools import chain
import json

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

os.chdir("/Users/alliesaizan/Documents/Python-Tinkering/Pudding")


##############################################
# Helper Functions

def get_pps(doc):
    """
    Function to get PPs from a parsed document. Sourced from:
    https://stackoverflow.com/questions/39100652/python-chunking-others-than-noun-phrases-e-g-prepositional-using-spacy-etc
    """
    
    pps = []
    for token in doc:
        # Try this with other parts of speech for different subtrees.
        if token.pos_ == 'ADP':
            pp = ' '.join([tok.orth_ for tok in token.subtree])
            pps.append(pp)
    return pps


def find_sentence_objects(tagged):
    """
    This function finds the direct objects in the article title.
    """
    try:
        objs = [i.text for i in tagged.noun_chunks if bool(re.search("dobj", i.root.dep_)) == True][0]
    except:
        objs = ""
    return(objs)


def find_verbs(doc):
    """
    This function is designed to pull verbs from sentences. It extracts the
    first verb because we only care about sentences where
    millennials are the subject of the sentence. It also removes helping verbs.
    """
    verbs = [i.lemma_ for i in doc if i.pos_ == "VERB"]

    # The list of the 23 auxillary (helping) verbs can be found here: https://en.wikipedia.org/wiki/Auxiliary_verb#List_of_auxiliaries_in_English
    helping_verbs = ["do", "does", "did", "has", "have", "had",\
                     "is", "am", "are", "was", "were", "be", "being",\
                     "been", "may", "must", "might", "should", "could",\
                     "would", "shall", "will","can"]
    
    if len(verbs) != 0:
        # Separating out helper verbs from the main verbs. Sometimes more than
        # one of these helper verbs can exist in a sentence, so I want to
        # ensure that I extract the main ("lexical") verb that reflects the subject's action.
        if verbs[0] in helping_verbs:
    
            modifiers = list( set(helping_verbs).intersection(set(verbs)) )
            #indicies = max([verbs.index(i) for i in modifiers])
            
            if len(modifiers) > 0 and any([i not in modifiers for i in verbs]):
                returnThis = [i for i in verbs if i not in modifiers][0]
            else: 
                returnThis = verbs[0]            
        else:    
            returnThis = verbs[0]
    else:
        # If the sentence does not contain any words tagged as verbs, return an empty string
        returnThis = ""
    return(returnThis)
            
    
def bs4_text_extraction(url):
    """
    This function uses the beautiful soup package to extract text from articles.
    It searches for paragraph ("p") tags, which are less likely to 
    be content we are not interested in (such as ads).
    """
    text = ""
    try:    
        res = requests.get(url, verify=False)
        soupText= bs4.BeautifulSoup(res.text, "lxml")
        for elem in soupText.find_all(['script']):
            elem.extract()
        selected = soupText.find_all('p', attrs={'class': None, 'script': None, 'span': None, 'style': None})
        for x in selected:
            text += x.getText()       
        return text
    except:
        return ""
    
def find_valences(l, analyzer):
    valences = [analyzer.polarity_scores(x)["compound"] for x in l]
    return pd.np.mean(valences)


##############################################
# Instantiate Event Registry API - commented out so I do not accidentally incur fees

api_key = "b3b5aa5d-a173-4102-97e6-227c795f7349"

er = EventRegistry(apiKey = api_key)

#q = QueryArticlesIter(
#    keywords = QueryItems.OR(["millennials", "Millennials", "millenial", "Millenial"]),
#    lang = "eng",
#    keywordsLoc="title",
#    dateStart = datetime.date(2015, 6, 16),
#    dateEnd = datetime.date(2019, 6, 1),
##    startSourceRankPercentile = 0,
##    endSourceRankPercentile = 20,
#    dataType = ["news"])

# The code below is how I saved the initial article pull - it is commented out but remains in the script to show my thought process.
#articles = pd.DataFrame(columns = ["title", "url", "text", "date"])

#for art in q.execQuery(er, sortBy = "date"):    
#    articles = articles.append({ "title": art["title"], "url": art["url"], "text": art["body"], "date": art["date"]}, ignore_index = True)

#pickle.dump(articles, open("articles.pkl", "wb"))

##############################################
# Cleaning and feature creation

articles = pickle.load(open("articles.pkl", "rb"))

# Export sample data
articles.sample(n = 100).to_csv("Sample Articles.csv", index = False)

articles["title"] = articles["title"].replace("&#\d+|\(|\)", "", regex = True)
articles["title"] = articles.title.apply(lambda x: re.split("\s*(\||;|\.|\s-\s)", str(x))[0])  # split on "|", ";","."

articles["title_lower"] = articles["title"].str.lower()

articles = articles[["title", "url", "text", "date", "title_lower", "tagged"]]
articles.drop_duplicates(inplace = True)

nlp = spacy.load("en_core_web_sm")
articles["tagged"] = articles["title"].apply(nlp)

articles["verbs"] = articles["tagged"].apply(find_verbs)

articles["objects"] = articles["tagged"].apply(find_sentence_objects)
articles["objects"] = articles["objects"].replace("^\s+", "", regex= True)

articles["subject"] = articles["tagged"].apply(lambda x: [token.text.lower() for token in x if token.dep_ in ["nsubj", "ROOT"]])
articles["mil_subj"] = articles["subject"].apply(lambda x: 1 if "millennial" in str(x).lower() else 0)

# Export the cleaned object
pickle.dump(articles, open("articles.pkl", "wb"))

# Extract articles about millennials
millennial_articles = articles.loc[articles["mil_subj"] == 1]

millennial_articles["split_text"] = millennial_articles["text"].apply(sent_tokenize)
millennial_articles["num_sentences"] = millennial_articles["split_text"].apply(len)
millennial_articles = millennial_articles.drop(labels = ["mil_subj", "split_text"], axis = 1)
millennial_articles = millennial_articles.loc[millennial_articles["num_sentences"] > 5]


##############################################
# Initial data exploration export to CSV
verbs = articles.verbs_text.tolist()
verbs = set(verbs)
verbs = list(verbs)

with open("verbs.txt", "w") as f:
    for verb in verbs:
        f.write("%s\n" % verb )
f.close()

objects = millennial_articles.objects.tolist()
objects = [i.lower() for i in list(chain.from_iterable(objects))]
objects = set(objects)
objects = list(objects)

with open("objects.txt", "w") as f:
    for obj in objects:
        f.write("%s\n" % obj )
f.close()

del verb, obj, verbs, objects


##############################################
# Group verbs by noun chunks and create the finalized JSON objects
millennial_articles["snippet"] = millennial_articles["text"].apply(lambda x: " ".join([i for i in x.split()][0:50]) )
millennial_articles["verbs"] = millennial_articles["verbs"].str.lower()
millennial_articles = millennial_articles.loc[millennial_articles["objects"] != ""]

articles_new = millennial_articles[["title_lower", "verbs", "objects"]].set_index(["title_lower","verbs"])["objects"].apply(pd.Series).stack()
articles_new = articles_new.reset_index()

articles_new = articles_new.drop(labels = "level_2", axis = 1).drop_duplicates()
articles_new.columns = ["title_lower", "verbs", "objects"]

articles_new["article_id"] = articles_new.index

articles_new["verbs"] = articles_new["verbs"].apply(lambda x: x.lower())
articles_new["objects"] = articles_new["objects"].apply(lambda x: x.lower())

# Start building the final nested JSON object

# Level 1: Verbs and their grouped objects
json_level1 = articles_new.loc[(articles_new.objects != "") & (articles_new.verbs != "")].groupby('verbs')['objects'].apply(set).reset_index()
json_level1["objects"] = json_level1["objects"].apply(list)

analyzer = SIA()
json_level1["avg_noun_valence"] = json_level1["objects"].apply(lambda x: find_valences(x, analyzer) )
json_level1.columns = ["verb", "nouns", "avg_noun_valence"]

# Level 2: Objects and their grouped verbs/articles
json_level2 = articles_new.loc[(articles_new["verbs"] != "") & (articles_new["objects"] != "")].groupby('objects')['verbs'].apply(set).reset_index()
json_level2["verbs"] = json_level2["verbs"].apply(list)

grouped_by_articles = articles_new[["title_lower", "objects"]].drop_duplicates()
grouped_by_articles = grouped_by_articles.groupby('objects')['title_lower'].apply(set).reset_index()
grouped_by_articles['title_lower'] = grouped_by_articles['title_lower'].apply(list) 

json_level2 = pd.merge(left = json_level2, right = grouped_by_articles, how = "left", on = "objects")
json_level2.columns = ["noun", "other_verbs", "articles"]

# Level 3: Article metadata
json_level3 = millennial_articles[["title_lower", "url", "date", "snippet", "verbs"]].drop_duplicates(subset = "title_lower")
json_level3["headline_valence"] = json_level3["title_lower"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
json_level3["article_tuple"] = list(zip(json_level3.title_lower, json_level3.verbs))
json_level3.drop(columns = "verbs", inplace = True)
json_level3.columns = ["headline", "url", "pub_date", "snippet", "headline_valence", "article_tuple"]

# Create the full nested JSON object

# Create the nested data frame from level 2 to level 3
tmpDataFrame = pd.DataFrame(columns= ["noun", "articles_dict", "avg_headline_valence"])
for index, row in json_level2.iterrows():
    valences = []
    headlines = []
    for headline in row["articles"]:
        tempdict = json_level3.loc[json_level3["headline"] == headline].to_dict("r")
        valences.append(json_level3.loc[json_level3["headline"] == headline, "headline_valence"].tolist())
        headlines.append(tempdict)
    valences = list(chain.from_iterable(valences))
    headlines = list(chain.from_iterable(headlines))
    tmpDataFrame = tmpDataFrame.append({"noun": row["noun"], "articles_dict" : headlines, "avg_headline_valence" : pd.np.mean(valences) }, ignore_index = True)
del tempdict, headline, valences, headlines, index, row

json_level2 = pd.merge(left = json_level2, right = tmpDataFrame, how = "left", on = "noun").drop(columns = ["articles"]).rename(columns = {"articles_dict":"articles"})

# Create the nested data frame from level 1 to level 2
tmpDataFrame = pd.DataFrame(columns = ["verb", "nouns_dict"])
for index, row in json_level1.iterrows():
    holder = []
    v = row["verb"]
    for noun in row["nouns"]:
        tempdf = json_level2.loc[json_level2["noun"] == noun]
        # Keep only articles that mention both the noun AND the verb (avoid repeating articles)
        tempdf["articles"] = tempdf["articles"].apply(lambda l_dicts: [i for i in l_dicts if v == i["article_tuple"][1] ])
        # Remove the verb from the list of "other verbs"
        tempdf["other_verbs"] = tempdf["other_verbs"].apply(lambda x: [i for i in x if i != v])
        tempdict = tempdf.to_dict("r")
        holder.append(tempdict)
    holder = list(chain.from_iterable(holder))
    tmpDataFrame = tmpDataFrame.append({"verb":row["verb"], "nouns_dict":holder}, ignore_index = True)
del tempdict, noun, holder, tempdf, index, row, v

new_json = pd.merge(left = json_level1, right = tmpDataFrame, how = "left", on = "verb").drop(columns = ["nouns"]).rename(columns = {"nouns_dict":"nouns"})
for_export = new_json.to_dict("r")

    
with open('articles_v2.json', 'w') as outfile:
    json.dump(for_export, outfile)
    
with open('articles_json.json', 'r') as f:
    testdat = json.load(f)


# Put together a list of nouns for export
included_verbs = pd.read_csv("verbs_to_include.csv")
tempdf = articles_new.loc[(articles_new.objects != "") & (articles_new.verbs != "")].groupby('verbs')['objects'].apply(set).reset_index()
tempdf["objects"] = tempdf["objects"].apply(list)

nouns = tempdf[tempdf["verbs"].isin(included_verbs["verbs"].tolist())]["objects"]
nouns = pd.DataFrame(set(chain.from_iterable(nouns.tolist())))
nouns.to_csv("included_nouns.csv", index = False)


##############################################
# Identify entries with missing articles

with open('articles_v2.json', 'r') as outfile:
    dat = json.load(outfile)

missings = pd.DataFrame(columns = ["verb", "noun"])

for item in dat:
    for level2 in item["nouns"]:
        if not level2["articles"]:
            missings = missings.append({"verb": item["verb"], "noun":level2["noun"]}, ignore_index = True)

##############################################
# Sentiment analysis - identifying the sentiment of the action verbs in sentences where millennials are the subject
analyzer = SIA()
millennial_articles["polarity"] = millennial_articles["text"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
negative_articles = millennial_articles.loc[millennial_articles["polarity"] < 0]

negative_articles[["title","text", "polarity"]].drop_duplicates().to_csv("Negative_Millennial_Articles.csv", index = False)

with open("verbs.txt", "w") as f:
    for verb in negative_verbs:
        f.write("%s\n" % verb )
f.close()

