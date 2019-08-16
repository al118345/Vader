from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

class Sentimiento:

    def fetch_sentiment_using_SIA(self,text):
        #python -m nltk.downloader vader_lexicon
        sid = SentimentIntensityAnalyzer()
        polarity_scores = sid.polarity_scores(text)
        if polarity_scores['neg'] > polarity_scores['pos']:
            return 'negative'
        else:
            return 'positive'

    def fetch_sentiment_using_textblob(self,text):
        analysis = TextBlob(text)
        # set sentiment
        if analysis.sentiment.polarity >= 0:
            return 'positive'
        else:
            return 'negative'
