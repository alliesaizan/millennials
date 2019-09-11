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
nltk.download('vader_lexicon')
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

def find_sentence_objects(tagged):
    """
    This function finds the direct objects in the article title.
    """
    try:
        objs = [i.text for i in tagged.noun_chunks if bool(re.search("dobj", i.root.dep_)) == True]
    except:
        objs = ""
    return(objs)

def findall(sub, lst, overlap = True):
    """
    This function finds the indicies where a sub-list occurs in a larger list.
    I adapted this function from:
    http://paddy3118.blogspot.com/2014/06/indexing-sublist-of-list-way-you-index.html
    """
    sublen = len(sub)
    firstthing = sub[0] if sub else []
    indices, indx = [], -1
    while True:
        try:
            indx = lst.index(firstthing, indx + 1)
        except ValueError:
            break
        if sub == lst[indx : indx + sublen]:
            indices.append(indx)
            if not overlap:
                indx += sublen - 1
    return(indices)

pattern = "(AUX\s)*(ADV\s)*(PART\s)*(VERB\s)+(ADV\s)*(PART\s)*"

def find_verb_phrases(title):
    """
    This function is designed to pull verb phrases from sentences where 
    millennials are the subject of the sentence. It assumes that the first 
    verb or verb phrase will refer to actions taken by millennials.
    """
    doc = nlp(title)
    
    # Obtain the parts of speech tags for each word in the title
    pos_tags = " ".join([i.pos_ for i in doc])
    
    # If the verb phrase pattern matches anywhere in the part of speech tags:
    if re.search(pattern, pos_tags): 
        # Find the matching tags and extract them as a list of tags.
        compiled = re.search(pattern, pos_tags)
        compare_this = compiled.group().split()
        # Compare this list of tags against the full list of tags for all the words in the title.
        # Extract the indicies where the tags occur in the list.
        result = findall(compare_this, pos_tags.split())[0]
        # In the title, pull out the words with matching indicies.
        verbs = " ".join([i.text for i in doc][result:result + len(compare_this)])
    else:
        # If the title does not contain any verb phrases, just extract the first verb in the sentence
        verbs = [i.text for i in doc if i.pos_ == "VERB"]
        if len(verbs) != 0:
            verbs = verbs[0]
        else:
            # If the sentence does not contain any words tagged as verbs, return an empty string
            verbs = ""
    return(verbs)


def find_verbs(title):
    """
    This function is designed to pull verbs from sentences. It extracts the
    first verb because we only care about sentences where
    millennials are the subject of the sentence.
    """    
    doc = nlp(title)
    
    verbs = [i.lemma_ for i in doc if i.pos_ == "VERB"]
    l_mods = ["be", "could", "can", "are", "have", "had", "were", \
              "been", "is", "will"]
    
    if len(verbs) != 0:
        # Separating out helper verbs from the main verbs. Sometimes more than
        # one of these helper verbs can exist in a sentence, so I want to
        # ensure that I extract the main verb that follows all those verbs.
        if verbs[0] in l_mods:
    
            modifiers = list( set(l_mods).intersection(set(verbs)) )
            indicies = max([verbs.index(i) for i in modifiers])
            
            if indicies + 1 < len(verbs):
                returnThis = verbs[indicies + 1]
            else: 
                returnThis = verbs[0]
        elif verbs[0] in ["must", "should"] and len(verbs) > 1:
            returnThis = verbs[1]
            
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
# Instantiate Event Registry API

api_key = "b3b5aa5d-a173-4102-97e6-227c795f7349"

er = EventRegistry(apiKey = api_key)

#q = QueryArticlesIter(
#    keywords = QueryItems.OR(["millennials", "Millennials", "millenial", "Millenial"]),
#    lang = "eng",
#    keywordsLoc="title",
#    dateStart = datetime.date(2015, 6, 16),
#    dateEnd = datetime.date(2015, 10, 11),
##    startSourceRankPercentile = 0,
##    endSourceRankPercentile = 20,
#    dataType = ["news"])

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

articles["title_lower"] = articles["title"].apply(lambda x: x.lower())

articles = articles[["title", "url", "text", "date", "title_lower", "tagged"]]
articles.drop_duplicates(inplace = True)

nlp = spacy.load("en_core_web_sm")
#articles["tagged"] = articles["title"].apply(nlp)

articles["verbs"] = articles["title"].apply(find_verbs)

articles["objects"] = articles["tagged"].apply(find_sentence_objects)
articles["objects"] = articles["objects"].replace("^\s+", "", regex= True)

articles["subject"] = articles["tagged"].apply(lambda x: [token.text.lower() for token in x if token.dep_ in ["nsubj", "ROOT"]])
articles["mil_subj"] = articles["subject"].apply(lambda x: 1 if "millennials" in str(x).lower() else 0)

# Find the publisher in the url domain
#publications = pd.read_csv("publications.csv")
#websites = publications["domain"].tolist()
#
#domains = [re.split("//(www\.)*", x)[-1] for x in websites]
#domains = [x.split(".")[0] for x in domains]
#
#articles["domain"] = articles["url"].apply(lambda x: re.split("//(www\.)*", x)[-1].split(".")[0])

# Export the cleaned object
pickle.dump(articles, open("articles.pkl", "wb"))

# Extract articles about millennials
millennial_articles = articles.loc[articles["mil_subj"] == 1]

millennial_articles["split_text"] = millennial_articles["text"].apply(sent_tokenize)
millennial_articles["num_sentences"] = millennial_articles["split_text"].apply(len)
millennial_articles = millennial_articles.drop(labels = ["mil_subj", "split_text"], axis = 1)

# Replacing the text of shorter articles (with a paragraph or less -- SKIP THIS)
short_arts = millennial_articles.loc[millennial_articles["num_sentences"] <= 5]
millennial_articles = millennial_articles.loc[millennial_articles["num_sentences"] > 5]
short_arts["text"] = short_arts["url"].apply(bs4_text_extraction)
millennial_articles.append(short_arts, ignore_index = True)
del short_arts


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
millennial_articles["objects_len"] = millennial_articles["objects"].apply(len)
millennial_articles.loc[millennial_articles.objects_len == 0, "objects"] = ""
millennial_articles["snippet"] = millennial_articles["text"].apply(lambda x: " ".join([i for i in x.split()][0:50]) )

articles_new = millennial_articles[["title_lower", "verbs", "objects"]].set_index(["title_lower","verbs"])["objects"].apply(pd.Series).stack()
articles_new = articles_new.reset_index()

articles_new = articles_new.drop(labels = "level_2", axis = 1).drop_duplicates()
articles_new.columns = ["title_lower", "verbs", "objects"]

articles_new["article_id"] = articles_new.index

articles_new["verbs"] = articles_new["verbs"].apply(lambda x: x.lower())
articles_new["objects"] = articles_new["objects"].apply(lambda x: x.lower())


#stemmer = SnowballStemmer("english")
#articles_new["verbs"] = articles_new["verbs"].apply(fix_be_verbs) 

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
json_level3 = millennial_articles[["title_lower", "url", "date", "snippet"]].drop_duplicates(subset = "title_lower")
json_level3["headline_valence"] = json_level3["title_lower"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
json_level3.columns = ["headline", "url", "pub_date", "snippet", "headline_valence"]

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
        tempdf["articles"] = tempdf["articles"].apply(lambda l_dicts: [i for i in l_dicts if v == find_verbs(i["headline"]) ])
        # Remove the verb from the list of "other verbs"
        tempdf["other_verbs"] = tempdf["other_verbs"].apply(lambda x: [i for i in x if i != v])
        tempdict = tempdf.to_dict("r")
        holder.append(tempdict)
    holder = list(chain.from_iterable(holder))
    tmpDataFrame = tmpDataFrame.append({"verb":row["verb"], "nouns_dict":holder}, ignore_index = True)
del tempdict, noun, holder, tempdf, index, row, v

json_level1 = pd.merge(left = json_level1, right = tmpDataFrame, how = "left", on = "verb").drop(columns = ["nouns"]).rename(columns = {"nouns_dict":"nouns"})
for_export = json_level1.to_dict("r")

# Removing extra brackets
#for_export = re.sub("\}\],\{noun", "\},\{noun", for_export)
#for_export = re.sub('\[\{\'noun\'', '\{noun', for_export)
#for_export = re.sub("\\\\", "", for_export)
    
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
# Sentiment analysis
analyzer = SIA()
millennial_articles["polarity"] = millennial_articles["text"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
negative_articles = millennial_articles.loc[millennial_articles["polarity"] < 0]

negative_articles[["title","text", "polarity"]].drop_duplicates().to_csv("Negative_Millennial_Articles.csv", index = False)

with open("verbs.txt", "w") as f:
    for verb in negative_verbs:
        f.write("%s\n" % verb )
f.close()


#pos_verbs = ["reviv", "using", "search", "driv", "consider", "attract", "balance",\
#             "refinanc", "bought", "hoard", "acquire", "purchase", "launch", "embrace",
#             "motivat","revamp", "save"]
#neg_verbs = ["messed", "delay", "bankrupt", "ignor", "criticize", "destroy", \
#         "offend", "recalculate", "scam", "crush", "fall", "rid", "outrag", \
#         "betray", "plummet", "divid", "prevent", "bash", "choos", "hurt", \
#         "hate", "eliminat", "lack", "wallow", "limit", "discourag", "ruin", \
#         "complain", "ban", "doom", "denounc", "refuse", "criticiz", \
#         "disgust", "doubt", "derail", "plague", "threaten", "offend", \
#         "detach", "delete", "disappear"]
