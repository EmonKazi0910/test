import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from transformers import pipeline
from bs4 import BeautifulSoup
from autoscraper import AutoScraper


# Load the sentiment analysis model from Hugging Face
sentiment_classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english') #Added New 

#########################################################################################################################

# Load the sentiment analysis model from Hugging Face
sentiment_classifier = pipeline('sentiment-analysis')
specific_model = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")  # 29 Nov 2023

def analyze_live_review():
    # Page title and description
    st.title("Live Review Analyzer")
    st.write("Enter a review in the text area below, and click the 'Analyze Review' button to get sentiment analysis.")

    # Input field for the user to enter a review
    live_review = st.text_area('Enter your review here', max_chars=1000)

    # Button to trigger sentiment analysis
    analyze_button = st.button('Analyze Review')

    # If the button is clicked and there is a review entered
    if analyze_button and live_review:
        # Perform sentiment analysis on the live review
        result = sentiment_classifier(live_review)[0]
        specific_result = specific_model(live_review)[0]  # Get result from specific model

        # Map sentiment label to 'Very Negative', 'Negative', 'Neutral', 'Positive', or 'Very Positive'
        sentiment_mapping = {'NEGATIVE': 'Negative', 'NEUTRAL': 'Neutral', 'POSITIVE': 'Positive'}
        sentiment_label = sentiment_mapping[result['label']]

        # Map specific model sentiment label to score
        specific_score = specific_result['score']
        specific_sentiment_label = sentiment_mapping[specific_result['label']]

        # Map score to sentiment category
        score_mapping = {1: 'Very Negative', 2: 'Negative', 3: 'Neutral', 4: 'Positive', 5: 'Very Positive'}
        score_category = score_mapping[int(round(specific_score * 4)) + 1]

        # Color code each analysis
        color_mapping = {'Negative': 'red', 'Neutral': 'yellow', 'Positive': 'green'}
        sentiment_color = color_mapping[sentiment_label]

        # Larger font for analysis
        st.subheader("Sentiment Analysis Result:")
        st.write(f"Review: {live_review}")
        st.write(f"Sentiment (General Model): {sentiment_label}")
        st.write(f"Confidence (General Model): {result['score']:.4f}")

        st.write(f"Sentiment (Specific Model): {specific_sentiment_label}")
        st.write(f"Score (Specific Model): {specific_score:.4f}")
        st.write(f"Category (Specific Model): {score_category}")

        # Apply color and larger font
        st.markdown(f"<p style='font-size:20px;color:{sentiment_color};'>{sentiment_label}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:20px;color:{sentiment_color};'>Confidence: {result['score']:.4f}</p>", unsafe_allow_html=True)


##############################################################################################################
# Function to perform sentiment analysis on Amazon reviews
def analyze_amazon_reviews():
    st.subheader("Amazon Review Scraper")
    amazon_url_input = st.text_input('Paste URL for Review Scraping')
    scrape_button = st.button('Scrape Reviews')

    if scrape_button:
        try:
            # ... (your existing code for Amazon review scraper)
            pass
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
###############################################################################################################
# Function to perform sentiment analysis on CSV file
def analyze_csv():
    st.subheader("Sentiment Analysis from CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Read the CSV file
                df = pd.read_csv(uploaded_file)

                # Display the first few rows of the dataset
                st.subheader("Dataset Preview:")
                st.write(df.head())

                # Confirm button after uploading CSV
                if st.button("Confirm CSV"):
                    # Perform sentiment analysis
                    st.subheader("Sentiment Analysis:")

                    # Get sentiment labels and scores
                    sentiment_results = sentiment_classifier(list(df['Review']))

                    # Extract sentiment labels from the results
                    sentiment_labels = [result['label'] for result in sentiment_results]

                    # Map sentiment labels to 'Positive', 'Negative', or 'Neutral'
                    sentiment_mapping = {
                        'NEGATIVE': 'Negative',
                        'NEUTRAL': 'Neutral',
                        'POSITIVE': 'Positive'
                    }
                    df['Sentiment_Label'] = [sentiment_mapping[label] for label in sentiment_labels]

                    # Display the total number of reviews
                    total_reviews = len(df)
                    st.subheader(f"Total Reviews: {total_reviews}")

                    # Display the counts for positive, negative, and neutral reviews
                    sentiment_counts = df['Sentiment_Label'].value_counts()
                    st.subheader("Sentiment Distribution:")
                    st.write(sentiment_counts)

                    # Display the results
                    st.subheader("Sentiment Results:")
                    st.write(df[['Review', 'Sentiment_Label']])

                    # Generate visualizations
                    st.subheader("Sentiment Distribution (Pie Chart):")
                    fig1, ax1 = plt.subplots()
                    ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
                    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                    st.pyplot(fig1)

                    st.subheader("Sentiment Distribution (Bar Chart):")
                    st.bar_chart(sentiment_counts)
                    pass
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
#############################################################################################################################
# Create a multi-page Streamlit app
def main():
    st.title('Sentiment Analysis App')

    # Create a sidebar with page selection
    menu = ["Home", "Live Review Analyzer", "Amazon Review Scraper", "Sentiment Analysis from CSV"]
    choice = st.sidebar.selectbox("Select Page", menu)

    if choice == "Home":
        st.subheader("Home Page")
        st.write("Welcome to the Sentiment Analysis App! Choose a page from the sidebar to get started.")
    elif choice == "Live Review Analyzer":
        analyze_live_review()
    elif choice == "Amazon Review Scraper":
        analyze_amazon_reviews()
    elif choice == "Sentiment Analysis from CSV":
        analyze_csv()

# Run the app
if __name__ == "__main__":
    main()
