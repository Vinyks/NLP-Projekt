import files
import nltk
import pandas as pd
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
import json
from gensim.models.coherencemodel import CoherenceModel
from gensim.utils import simple_preprocess
from gensim.models import LdaModel


if __name__ == '__main__':

    nltk.download('stopwords')
    nltk.download('punkt_tab')

    #Datenset über pandas einlesen
    df = pd.read_csv('dataset.csv', encoding='utf-8')

    #Liste an Namen von Spielen die von Valve produziert wurden
    valveGames = ["Portal 2", "Left 4 Dead", "Portal", "Counter-Strike",
                  "Counter-Strike: Source", "Half-Life", "Team Fortress Classic", "Day of Defeat: Source",
                  "Day of Defeat", "Ricochet", "Deathmatch Classic",
                  "Half-Life 2: Deathmatch", "Half-Life Deathmatch: Source", "Alien Swarm",
                  "Counter-Strike: Condition Zero", "Half-Life: Source",
                  "Half-Life 2", "Half-Life 2: Lost Coast", "Half-Life 2: Episode One", "Half-Life 2: Episode Two",
                  "Left 4 Dead 2", "Dota 2", "The Lab", "Team Fortress 2"]

    def remove_stopwords(text):
        words = text.split()
        result_words = [word for word in words if word not in stopwords.words('english')]
        text = " ".join(result_words)

        return text

    #Verwandelt Text in Kleinbuchstaben und entfernt Stoppwörter
    def preprocess(text):
        text = text.lower()
        text = remove_stopwords(text)

        return text

    #Funktion liest aus der CSV Datei die Bewertungen, vorverarbeitet sie und schreibt sie in eine Liste
    def extract_reviews(df):
        reviews = []
        for i, row in df.iterrows():

            review = row['review_text']

            #Reviews die keine zeichenkette sind rausfiltern
            if (type(review) != str or row['app_name'] not in valveGames):
                continue

            review = preprocess(row['review_text'])
            reviews.append(review)
            print(i)
        return reviews

    def write_reviews_to_file(reviews):
        with open("reviews.json", "w") as final:
            json.dump(reviews, final)

    def get_best_topic_amount(reviews, max_topics, min_topics, stepsize):
        #Bewertungen Tokenizen
        tokenized_reviews = [simple_preprocess(review) for review in reviews]
        #Wörterbuch erstellen
        id2word = Dictionary(tokenized_reviews)
        #Korpus über Tokenisierte Bewertungen erstellen
        corpus = [id2word.doc2bow(review) for review in tokenized_reviews]

        #choerence score liste
        scores = []

        #LDA Modelle über Gensim erstellen und choerence_scores speichern
        for i in range(1, int((max_topics-min_topics)/stepsize)):
            topics = min_topics+(i * stepsize)

            lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=topics, passes=5, iterations=50,
                                 chunksize=2000, random_state=42)

            coherence_model = CoherenceModel(model=lda_model, texts=tokenized_reviews, dictionary=id2word,
                                             coherence='c_v', processes=1)
            coherence_score = coherence_model.get_coherence()

            scores.append(coherence_score)

            print("For ", topics, " Topics achieved score of: ", coherence_score)


        #Beste Themenanzahl finden
        highest = 0
        index = -1
        for i in range(0, len(scores)):
            if scores[i] > highest:
                highest = scores[i]
                index = i
        best_topic_amount = index*stepsize+min_topics+1

        print("Highes score for: " + str(best_topic_amount) + " Topics, with score of " + str(highest))

        #Über gensim erneut mit der besten Themenanzahl das Modell trainieren
        lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=best_topic_amount, passes=5, iterations=50,
                             chunksize=2000, random_state=42)

        #Ausgabe der 20 top Wörter der Themen
        topics = lda_model.print_topics(num_words=20)
        for topic_id, topic_words in topics:
            words_only = []
            for word_prob in topic_words.split(" + "):
                word = word_prob.split("*")[1].strip('"')
                words_only.append(word)

            print(f"\n Topic {topic_id}:")
            print(f"   {', '.join(words_only)}")

        return best_topic_amount


    print("Corpus Length: " + str(len(df)))

    #reviews = extract_reviews(df)

    #write_reviews_to_file(reviews)

    #Bewertungen über Datei einlesen
    with open("reviews.json", "r") as final:
        reviews = json.load(final)


    #Anzahl an Bewertungen auf 5000 Beschränken für Testzwecke
    trimmedReviews = []
    for i, review in enumerate(reviews):
        if i > 5000:
            break
        trimmedReviews.append(review)
    reviews = trimmedReviews

    #Beste Themenanzahl herausfinden
    best_topic_amount = get_best_topic_amount(reviews, 20, 10, 1)

    #BoW Vektor erstellen und füttern
    bowVectorizer = CountVectorizer(max_features=50)
    bow = bowVectorizer.fit_transform(reviews)
    bow = pd.DataFrame(bow.toarray(), columns=bowVectorizer.get_feature_names_out())

    print(bow)

    #Tfidf Vektor erstellen und füttern
    tfidfVectorizer = TfidfVectorizer(use_idf=True, max_features=50,smooth_idf=True)
    tfidf = tfidfVectorizer.fit_transform(reviews)
    tfidf = pd.DataFrame(tfidf.toarray(), columns=tfidfVectorizer.get_feature_names_out())

    print(tfidf) 

    #LSA Modell erstellen und trainieren
    LSA_model = TruncatedSVD(n_components=best_topic_amount, algorithm='randomized', n_iter=10)
    lsa = LSA_model.fit_transform(tfidf)

    #Ausgeben des LSA Ergebnisses
    print("Review 1 LSA: ")
    for i,topic in enumerate(lsa[1]):
        print("Topic ",i,": ",topic*100)

    #LDA Modell erstellen und trainieren
    lda_model = LatentDirichletAllocation(n_components=best_topic_amount,learning_method='online', random_state=42, max_iter=5)
    lda_top = lda_model.fit_transform(tfidf)

    #Ausgeben des LDA Ergebnisses
    print("Review 1 LDA: ")
    for i,topic in enumerate(lda_top[1]):
        print("Topic ",i,": ",topic*100, "%")