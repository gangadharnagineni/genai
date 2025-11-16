import time
import torch
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re


class CustomerFeedbackAnalyzer:

    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment"):

        print("Loading sentiment model...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def preprocess_dataset(self, df, text_col):
        df = df.copy()
        df[text_col] = df[text_col].astype(str)
        df[text_col] = df[text_col].fillna("")
        df[text_col] = df[text_col].apply(self.clean_text)

        processed_list = df[text_col].tolist()
        return processed_list

    def get_sentiment(self, text):
        tokens = self.tokenizer(text, return_tensors="pt",
                                truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**tokens)

        probs = outputs.logits.softmax(dim=1).cpu().numpy()[0]
        label = {0: "negative", 1: "neutral", 2: "positive"}[probs.argmax()]
        return label

    def apply_sentiment(self, texts):
        sentiments = []
        for t in texts:
            sentiments.append(self.get_sentiment(t))
        return sentiments

    def extract_topics(self, texts, num_topics=10, num_words=10):

        if len(texts) < 20:
            return ["Not enough reviews for topic modeling"]

        vectorizer = TfidfVectorizer(stop_words="english", max_df=0.95, min_df=2)
        X = vectorizer.fit_transform(texts)

        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(X)

        feature_names = vectorizer.get_feature_names_out()
        topics = []

        for idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[-num_words:]]
            topics.append(f"Topic {idx+1}: " + ", ".join(top_words))

        return topics

    def performance_test(self, func, *args):
        start = time.time()
        result = func(*args)
        end = time.time()
        print(f"Processing time: {round(end-start, 2)} seconds")
        return result


if __name__ == "__main__":

    df = pd.read_csv("C://Users//saiad//Downloads//reviews.csv")
    text_col = "review"

    analyzer = CustomerFeedbackAnalyzer()

    # CLEAN & LIST
    reviews = analyzer.preprocess_dataset(df, text_col)

    # SENTIMENT
    sentiments = analyzer.apply_sentiment(reviews)
    df["sentiment"] = sentiments

    df_positive = df[df["sentiment"] == "positive"]
    df_negative = df[df["sentiment"] == "negative"]

    print("\nPositive Topics:")
    positive_topics = analyzer.extract_topics(df_positive[text_col].tolist(), 10, 10)
    for t in positive_topics:
        print(t)

    print("\nNegative Topics:")
    negative_topics = analyzer.extract_topics(df_negative[text_col].tolist(), 10, 10)
    for t in negative_topics:
        print(t)

    df.to_csv("processed_reviews.csv", index=False)
    print("done.")
