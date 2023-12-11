# Import necessary libraries
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from bs4 import BeautifulSoup
from autoscraper import AutoScraper
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import urllib.request
import numpy as np
import plotly.express as px
from scipy.special import softmax
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
from selenium.common.exceptions import NoSuchElementException 

###################################CODE#########################################################################

# Load the sentiment analysis model from Hugging Face
@st.cache_resource
def load_sentiment_model():
    # Load tokenizer and model for sentiment analysis
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    return tokenizer, model

def load_sentiment_modelurl():
    # Load tokenizer and model for sentiment analysis from a different source
    tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    return tokenizer, model

def load_sentiment_modelcsv():
    # Load sentiment analysis model using pipeline (assuming from transformers library)
    sentiment_model = pipeline('sentiment-analysis')
    return 

###############################################################################################################################

# Live Review Analyzer
def analyze_live_review():
    # Streamlit UI for live review analysis
    st.subheader("Live Review Analyzer")
    tokenizer, model = load_sentiment_model()
    live_review = st.text_area('Enter your review here', max_chars=1000)
    analyze_button = st.button('Analyze Review')

    if analyze_button and live_review:
        # Tokenize and analyze live review
        inputs = tokenizer(live_review, return_tensors='pt')
        outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits)
        score = torch.softmax(outputs.logits, dim=1).max()
        sentiment_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        sentiment_label = sentiment_mapping[predicted_label.item()]

        # Display sentiment analysis result
        st.subheader("Sentiment Analysis Result:")
        st.write(f"Review: {live_review}")
        st.write(f"Sentiment: {sentiment_label}")
        st.write(f"Confidence: {score.item():.4f}")

###############################################################################################################

# Function to perform sentiment analysis on Amazon reviews
def analyze_amazon_reviews():
    tokenizer, model = load_sentiment_modelurl()

    # Naming the Web App
    st.title('Sentiment 2.0')
    st.markdown('This module uses web scraping to get product reviews from the provided Amazon URL. '
                'It then processes the reviews through the HuggingFace transformers model for sentiment analysis.'
                'The resulting sentiments and corresponding reviews are, '
                'then put in a dataframe for display which is what you see as a result.')

    # Web scraping function to get product reviews from the provided Amazon URL
    def scrape_amazon_reviews(url, num_reviews=5):
        driver = webdriver.Chrome()
        reviews_data = []

        try:
            # Navigate to the initial URL and click "See more reviews"
            driver.get(url)
            see_more_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, '//a[@data-hook="see-all-reviews-link-foot"]'))
            )
            see_more_button.click()

            while True:
                # Wait for the page to load
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, '//div[@class="a-row a-spacing-small review-data"]'))
                )

                # Extract product reviews based on the HTML structure
                review_elements = driver.find_elements(By.XPATH, '//div[@class="a-row a-spacing-small review-data"]')

                # Get the specified number of reviews
                for i, review_element in enumerate(review_elements[:num_reviews]):
                    review_text_element = review_element.find_element(By.XPATH, './/span[@data-hook="review-body"]')
                    if review_text_element:
                        review_text = review_text_element.find_element(By.XPATH, './/span').text.strip()
                        reviews_data.append(f'Review {i + 1}: {review_text}')

                # Check if the number of scraped reviews is equal to or exceeds the requested number
                if len(reviews_data) >= num_reviews:
                    break

                # Check if the "Next page" button is clickable
                try:
                    next_page_button = driver.find_element(By.XPATH, '//a[contains(@href, "pageNumber")][contains(text(), "Next page")]')
                except NoSuchElementException:
                    break  # Break the loop if the "Next page" button is not found

                # Click the "Next page" button
                next_page_button.click()

                # Add a delay before the next iteration to ensure the page has fully loaded
                time.sleep(3)  # You can adjust the delay time as needed

        finally:
            driver.quit()

        return reviews_data

    # Sentiment analysis function using DistilBERT
    def distilbert_sentiment_analysis(reviews):
        # Tokenize the reviews
        inputs = tokenizer(reviews, return_tensors="pt", truncation=True, padding=True)

        # Forward pass through the model
        outputs = model(**inputs)

        # Extract the predicted labels
        predicted_labels = torch.argmax(outputs.logits, dim=1)

        return predicted_labels.tolist()

    # MAIN
    def run():
        # Create a form in Streamlit
        with st.form(key='Enter product'):
            # Input field for the user to enter the Amazon URL
            amazon_url_input = st.text_input('Enter the Amazon URL for the product')

            # Input field for the user to enter the number of reviews to analyze (with a default of 5)
            num_reviews = st.number_input('Enter the number of reviews for which you want to know the sentiment', 1, 50, 5)

            # Submit button to trigger the form submission
            submit_button = st.form_submit_button(label='Submit')

            # Check if the form has been submitted
            if submit_button:
                # Use web scraping to fetch product reviews from the provided Amazon URL
                reviews = scrape_amazon_reviews(amazon_url_input, num_reviews)

                # Use the 'distilbert_sentiment_analysis' function to classify sentiments of the reviews
                q = distilbert_sentiment_analysis(reviews)

                # Create a DataFrame to display the results
                df = pd.DataFrame(list(zip(reviews, q)), columns=['Product Reviews', 'sentiment'])
                df2 = pd.DataFrame(
                    {"Reviews": reviews, "Rating out of 5": q}
                )

                # Display the DataFrame using Streamlit
                st.write(df)
                st.bar_chart(df2, x="Rating out of 5", y="Reviews")

    # Run the Streamlit app if the script is executed directly
    if __name__ == '__main__':
        run()

################################################################################################################

# Load the sentiment analysis model from Hugging Face
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Cache AutoTokenizer
@st.cache_resource(max_entries=1, hash_funcs={AutoTokenizer: lambda _: None})
def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)

tokenizer = load_tokenizer()

# Cache AutoModelForSequenceClassification
@st.cache_resource(max_entries=1, hash_funcs={AutoModelForSequenceClassification: lambda _: None})
def load_model():
    return AutoModelForSequenceClassification.from_pretrained(MODEL)

model = load_model()

# Function to preprocess text
def preprocess(text):
    # Preprocess text by replacing user mentions and URLs
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Function to perform sentiment analysis
@st.cache_resource(max_entries=1, hash_funcs={pd.DataFrame: lambda _: None})
def analyze_sentiment(df):
    # Perform sentiment analysis on the provided DataFrame
    text_column = 'Review text' if 'Review text' in df.columns else 'Reviews'
    texts = df[text_column].tolist()

    scores_list = []

    for text in texts:
        preprocessed_text = preprocess(text)
        encoded_input = tokenizer(preprocessed_text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        scores_list.append(scores)

    return scores_list

# Function to perform sentiment analysis on CSV data
def analyze_csv():
    st.subheader("Sentiment Analysis from CSV")

    upl = st.file_uploader('Upload CSV file', type=["csv"])

    if upl:
        df = pd.read_csv(upl, encoding='latin1')

        scores_list = analyze_sentiment(df)

        labels = np.argmax(scores_list, axis=1)
        df['Sentiment_Label'] = labels

        sentiment_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        df['Sentiment_Label'] = df['Sentiment_Label'].map(sentiment_mapping)

        st.subheader("Number of Reviews")
        st.write(len(df))

        st.subheader("Sentiment of Reviews")
        st.write(df['Sentiment_Label'].value_counts())

        # Pie chart showing the distribution of sentiments
        st.subheader("Pie Chart - Sentiment Distribution")
        fig = px.pie(df, names='Sentiment_Label', title='Sentiment Distribution')
        st.plotly_chart(fig)

#####################################################################################################################

# Create a multi-page Streamlit app
def main():
    st.title('Sentiment Analysis Web App')

    # Sidebar menu for different pages
    menu = ["Home", "Live Review Analyzer", "Amazon Review Scraper", "Sentiment Analysis from CSV"]
    choice = st.sidebar.selectbox("Select Page", menu)

    if choice == "Home":
        # Home page with welcome message
        st.subheader("Welcome")
        st.write("Explore the power of sentiment analysis with My innovative project. This web app empowers you to understand and analyze sentiments effortlessly. Dive into the world of customer feedback and reviews, gaining valuable insights for informed decisions. Discover the future of sentiment analysis, crafted with precision for your convenience.")
    elif choice == "Live Review Analyzer":
        # Call the live review analyzer function
        analyze_live_review()
    elif choice == "Amazon Review Scraper":
        # Call the Amazon review scraper function
        analyze_amazon_reviews()
    elif choice == "Sentiment Analysis from CSV":
        # Call the CSV sentiment analysis function
        analyze_csv()

if __name__ == "__main__":
    main()

# Remove Streamlit default footer
st.markdown("""
    <style>
        .viewerBadge_container__1QSob {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# Add your custom footer with small and grey styling
st.markdown(
    "<p style='font-size: 10px; color: grey;'>Made By Emon Kazi Abu Taleb (192401) - Arab Open University</p>",
    unsafe_allow_html=True
)
