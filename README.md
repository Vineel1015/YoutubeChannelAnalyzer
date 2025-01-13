# YouTube Channel Performance Analyzer

This project analyzes YouTube channels and evaluates their performance based on key metrics such as view count, like count, and comment count. By processing and analyzing these metrics, the project provides insights into video and channel performance, including projections of future growth.

#Features

## Key Metrics Analysis:
Extracts and evaluates three key performance metrics: view count, like count, and comment count for videos in a given channel.
### Performance Quantification:
Calculates a like-to-view ratio for each video and compares it against other videos in the channel to classify whether it is overperforming or underperforming.
### Channel Strength Projection:
Determines the overall channel strength by analyzing the proportion of overperforming videos.
Channels with a majority of overperforming videos are classified as "strong" and projected to experience growth.
### Data Cleaning and Preprocessing:
Removes outliers and ensures clean data for accurate analysis.
Applies statistical methods such as Z-score normalization to handle anomalies.
### Machine Learning Pipeline:
Trains a classification model to predict video performance based on input features: view count, like count, and comment count.
Implements preprocessing techniques such as scaling, power transformations, and logistic regression to ensure robust predictions.
### YouTube API Integration:
Extracts video data directly from YouTube using the YouTube Data API v3, including video IDs, statistics, and metadata.

# Files Overview

# Data_Extraction.ipynb

## This notebook handles the extraction of data from the YouTube Data API, including:

Fetching video IDs from a specified YouTube channel.
Extracting video metadata and performance statistics (view count, like count, comment count).
Storing the extracted data in a structured format for further analysis.

# Data_Cleaning_Visualization-2.ipynb

## This notebook focuses on:

Cleaning and preprocessing the extracted data, including outlier removal and normalization.
Implementing a machine learning pipeline to classify video performance.
Visualizing insights, such as trends in channel performance and the distribution of key metrics.

# sam_videos.csv

## A sample dataset containing performance metrics for videos in a YouTube channel. The dataset includes:

Video IDs.
Category IDs.
View count, like count, and comment count for each video.

# How It Works

Data Extraction:
Use the Data_Extraction.ipynb notebook to pull data from a specified YouTube channel. The API fetches video statistics, including view count, like count, and comment count.
Data Cleaning and Preprocessing:
The Data_Cleaning_Visualization-2.ipynb notebook removes data anomalies (e.g., outliers) and calculates additional metrics like the like-to-view ratio.
Performance Classification:
Videos are classified as overperforming or underperforming based on their like-to-view ratio compared to the channel's median.
Channel Analysis:
A strong channel is identified by the percentage of overperforming videos.
Channels classified as strong are deemed likely to grow, based on video performance trends.

# Future Scope

Add additional performance metrics, such as watch time and subscriber growth, for a more comprehensive analysis.
Develop a web-based interface for real-time data visualization and user interaction.
Incorporate advanced machine learning models to predict video performance based on additional metadata.

# Prerequisites

Python 3.x
Required Python Libraries:
pandas, numpy, matplotlib, seaborn
scikit-learn for machine learning
googleapiclient for YouTube API integration

# Usage

Set up your API key for the YouTube Data API in the Data_Extraction.ipynb notebook.
Extract data for your desired YouTube channel.
Clean and preprocess the data using the Data_Cleaning_Visualization-2.ipynb notebook.
Analyze channel performance and review projections.
