import files
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
nltk.download('stopwords')
nltk.download('punkt_tab')
import json

df = pd.read_csv('dataset.csv', encoding='utf-8')

valveGames = ["Portal 2", "Left 4 Dead", "Portal", "Counter-Strike",
              "Counter-Strike: Source", "Half-Life", "Team Fortress Classic", "Day of Defeat: Source",
              "Day of Defeat", "Ricochet", "Deathmatch Classic",
              "Half-Life 2: Deathmatch", "Half-Life Deathmatch: Source", "Alien Swarm",
              "Counter-Strike: Condition Zero", "Half-Life: Source",
              "Half-Life 2", "Half-Life 2: Lost Coast", "Half-Life 2: Episode One", "Half-Life 2: Episode Two",
              "Left 4 Dead 2", "Dota 2", "The Lab", "Team Fortress 2"]

vocabulary = []

def preprocess(text):
    text = text.lower()

    text = remove_stopwords(text)

    return text

def remove_stopwords(text):

    words = text.split()
    result_words = [word for word in words if word not in stopwords.words('english')]
    text = " ".join(result_words)

    return text

def extract_reviews(df):
    reviews = []
    for i, row in df.iterrows():

        review = row['review_text']

        # Filter wrongly typed reviews
        if (type(review) != str):
            continue

        review = preprocess(row['review_text'])
        reviews.append(review)
        print(i)
    return reviews

def write_reviews_to_file(reviews):
    with open("reviews.json", "w") as final:
        json.dump(reviews, final)

print("Corpus Length: " + str(len(df)))

#reviews = extract_reviews(df)

#write_reviews_to_file(reviews)

with open("reviews.json", "r") as final:
    reviews = json.load(final)

bowVectorizer = CountVectorizer(max_features=500)
bow = bowVectorizer.fit_transform(reviews)
bow = pd.DataFrame(bow.toarray(), columns=bowVectorizer.get_feature_names_out())

print(bow)

tfidfVectorizer = TfidfVectorizer(use_idf=True, max_features=500,smooth_idf=True)
tfidf = tfidfVectorizer.fit_transform(reviews)
tfidf = pd.DataFrame(tfidf.toarray(), columns=tfidfVectorizer.get_feature_names_out())

print(tfidf)

LSA_model = TruncatedSVD(n_components=100, algorithm='randomized', n_iter=10)
lsa = LSA_model.fit_transform(tfidf)

print("Review 1: ")
for i,topic in enumerate(lsa[1]):
    print("Topic ",i,": ",topic*100)


lda_model = LatentDirichletAllocation(n_components=100,learning_method='online', random_state=42, max_iter=5)
lda_top = lda_model.fit_transform(tfidf)

print("Review 1: ")
for i,topic in enumerate(lda_top[1]):
    print("Topic ",i,": ",topic*100,"%")

