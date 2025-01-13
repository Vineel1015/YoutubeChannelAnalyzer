from flask import Flask, request, jsonify
from model_pipeline import train_model, analyze_channel, fetch_channel_data, load_and_clean_data
import pandas as pd
import os
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

API_KEY = os.getenv("YOUTUBE_API_KEY")
if not API_KEY:
    raise ValueError("Missing API key. Set the YOUTUBE_API_KEY environment variable.")
try:
    data = load_and_clean_data(fetch_channel_data(API_KEY, "UCtUbO6rBht0daVIOGML3c8w"))
    model, accuracy = train_model(data)
    logging.info(f"Default model trained with mock data. Accuracy: {accuracy}")
except Exception as e:
    logging.error(f"Failed to initialize default model: {e}")
    model = None

@app.route('/analyze', methods=['GET'])
def analyze():
    channel_id = request.args.get('channelId')
    if not channel_id:
        return jsonify({"message": "Channel ID is required"}), 400

    try:
        channel_data = fetch_channel_data(API_KEY, channel_id)
        required_columns = ["View Count", "Like Count", "Comment Count"]
        if channel_data.empty or not all(col in channel_data.columns for col in required_columns):
            return jsonify({"message": "Failed to retrieve sufficient data for the channel."}), 400
        analysis_results = analyze_channel(model, channel_data)

        return jsonify(analysis_results)

    except Exception as e:
        logging.error(f"Error during channel analysis: {e}")
        return jsonify({"message": "An error occurred while analyzing the channel."}), 500

if __name__ == "__main__":
    app.run(debug=True)
