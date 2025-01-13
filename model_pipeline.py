import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from scipy import stats
from googleapiclient.discovery import build

def load_and_clean_data(data):
    def remove_outliers_zscore(data, column, threshold=3):
        z_scores = np.abs(stats.zscore(data[column]))
        return data[(z_scores < threshold)]

    df = remove_outliers_zscore(data, "View Count")
    df["LikesToViewsRatio"] = df["Like Count"] / data["View Count"].replace(0, np.nan)
    median_ratio = df["LikesToViewsRatio"].median()
    df["PerformedBetterThanAverage"] = (df["LikesToViewsRatio"] > median_ratio).astype(int)
    return df.dropna()

def fetch_channel_data(api_key, channel_id):
    youtube = build("youtube", "v3", developerKey=api_key)
    video_ids = []
    video_data = []

    try:
        request = youtube.channels().list(id=channel_id, part="contentDetails")
        response = request.execute()
        upload_playlist_id = response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
        next_page_token = None
        while True:
            request = youtube.playlistItems().list(
                playlistId=upload_playlist_id,
                part="snippet",
                maxResults=50,
                pageToken=next_page_token
            )
            response = request.execute()
            video_ids.extend(
                item["snippet"]["resourceId"]["videoId"] for item in response["items"]
            )
            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
        for i in range(0, len(video_ids), 50):
            request = youtube.videos().list(
                id=",".join(video_ids[i:i + 50]),
                part="statistics"
            )
            response = request.execute()
            for item in response["items"]:
                stats = item["statistics"]
                video_data.append({
                    "Video ID": item["id"],
                    "View Count": int(stats.get("viewCount", 0)),
                    "Like Count": int(stats.get("likeCount", 0)),
                    "Comment Count": int(stats.get("commentCount", 0)),
                })

    except Exception as e:
        print(f"Error fetching channel data: {e}")
    
    return pd.DataFrame(video_data)

def train_model(data):
    X = data[["View Count", "Like Count", "Comment Count"]]
    y = data["PerformedBetterThanAverage"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), ["View Count", "Like Count", "Comment Count"])]
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression())
    ])
    pipeline.fit(X_train, y_train)
    
    accuracy = pipeline.score(X_test, y_test)
    return pipeline, accuracy

def analyze_channel(model, data):
    avg_views = data["View Count"].mean()
    avg_likes = data["Like Count"].mean()
    avg_comments = data["Comment Count"].mean()
    
    predictions = model.predict(data[["View Count", "Like Count", "Comment Count"]])
    overperforming_count = sum(predictions)
    total_videos = len(predictions)

    return {
        "total_videos": total_videos,
        "average_views": avg_views,
        "average_likes": avg_likes,
        "average_comments": avg_comments,
        "overperforming_videos": overperforming_count,
        "channel_performance": "Strong" if overperforming_count / total_videos > 0.5 else "Weak"
    }
