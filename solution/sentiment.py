from nltk.sentiment import SentimentIntensityAnalyzer
import snscrape.modules.twitter as sntwitter

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Define a function to calculate the sentiment score for each tweet
def calculate_sentiment_score(tweet):
    return sia.polarity_scores(tweet)['compound']

# Define a function to scrape tweets related to Bitcoin for a given date range
def scrape_tweets(query, start_date, end_date):
    tweets = []
    query_str = f'{query} since:{start_date} until:{end_date}'
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query_str).get_items()):
        tweets.append(tweet.content)
    return tweets

# Define a function to calculate the sentiment figure for each day
def calculate_sentiment_figure(tweets):
    sentiment_figure = []
    for tweet in tweets:
        sentiment_score = calculate_sentiment_score(tweet)
        sentiment_figure.append(sentiment_score)
    return sentiment_figure

# Define a function to get the sentiment figure for each day in a date range
def get_sentiment_figure(query, start_date, end_date):
    tweets = scrape_tweets(query, start_date, end_date)
    sentiment_figure = calculate_sentiment_figure(tweets)
    return sentiment_figure

# Provide the necessary inputs and call the get_sentiment_figure function to get the sentiment figure for each day
query = 'Bitcoin'
start_date = '2023-05-01'
end_date = '2023-05-10'

sentiment_figure = get_sentiment_figure(query, start_date, end_date)
print(sentiment_figure)
