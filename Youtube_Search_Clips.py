import openai
from rake_nltk import Rake
from googleapiclient.discovery import build
import cv2
import pytube
import os
from multiprocessing import Pool
from moviepy.editor import VideoFileClip
import nltk
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import base64
import tensorflow as tf
import torch
import tokenizer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



import torch
from torchvision import transforms
#from frame_analysis import analyze_frame

# Define the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")



def preprocess_image(image_path):
    """
    Load and preprocess an image.
    """
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_tensor = torch.tensor(img_array, dtype=torch.float32)
    img_tensor = img_tensor.permute(0, 3, 1, 2)  # Swap channels and height/width dimensions
    return img_tensor


def generate_question(keywords, frame_index):
    question = f"What is happening in frame {frame_index}?"
    for keyword in keywords:
        question += f" And what is the {keyword} doing?"
    return question



def analyze_frame(frame, frame_index, keywords, tokenizer, model):
    # Convert the frame to a PIL image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Define the transform to preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Preprocess the image and construct the question using the keywords
    input_image = transform(pil_image)
    question = f"What is happening in this image? {', '.join(keywords)}"

    # Use the model to answer the question and get the confidence score
    inputs = tokenizer(question, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]
    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
        confidence_score = torch.max(answer_start_scores)

    # Return the original frame and confidence score
    return frame, confidence_score

# Set up OpenAI API key
openai.api_key = "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"

# Set up Rake keyword extractor
rake = Rake()

# Get input from user
input_text = input("Enter your group of words: ")

# Extract phrases and keywords
phrases = []
for sentence in input_text.split("."):
    rake.extract_keywords_from_text(sentence)
    phrases.append(" ".join(rake.get_ranked_phrases()[:2]))

# Initialize YouTube API client
youtube = build('youtube', 'v3', developerKey='AIzaSyBuFdwbkHbckk1sEOp_vbK9sb6QYEhJCTg')

for phrase in phrases:
    print(f"Searching for videos related to '{phrase}'...")
    video_id = None
    video_title = None
    video_description = None
    video_thumbnail_url = None

    # Try a specific search first
    search_response = youtube.search().list(
        q=phrase,
        type='video',
        videoDefinition='high',
        videoDuration='medium',
        part='id,snippet',
        fields='items(id(videoId),snippet(channelId,publishedAt,channelTitle,title,description,thumbnails(default)))'
    ).execute()

    if len(search_response['items']) == 0:
        print(f"No videos found for phrase '{phrase}'. Trying a more generalized search...")
        search_response = youtube.search().list(
            q=phrase.split()[0],
            type='video',
            videoDefinition='high',
            videoDuration='medium',
            part='id,snippet',
            fields='items(id(videoId),snippet(channelId,publishedAt,channelTitle,title,description,thumbnails(default)))'
        ).execute()

    if len(search_response['items']) > 0:
        video_id = search_response['items'][0]['id']['videoId']
        video_title = search_response['items'][0]['snippet']['title']
        video_description = search_response['items'][0]['snippet']['description']
        video_thumbnail_url = search_response['items'][0]['snippet']['thumbnails']['default']['url']

    if video_id is None:
        print(f"No videos found for phrase '{phrase}'")
        continue

    # Use OpenAI to analyze the video description for relevant keywords
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Extract relevant keywords from the following video description to make the search more relevant: {video_description} {video_title}",
        max_tokens=30,
        n=1,
        stop=None,
        temperature=0.5,
    )
    keywords = [keyword.strip() for keyword in response.choices[0].text.split("\n") if keyword]

    # Download the video using pytube
    print(f"Downloading video '{video_title}'...")
    video = pytube.YouTube(f"https://www.youtube.com/watch?v={video_id}")
    video_clip = video.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    video_clip.download(output_path='downloads', filename=phrase.replace(' ', '_') + '.mp4')

    # Use visual question answering to analyze frames
    cap = cv2.VideoCapture(os.path.join('downloads', phrase.replace(' ', '_') + '.mp4'))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    print(f"Analyzing frames for video '{video_title}'...")
    for i in range(0, frame_count, 10):  # Skip every 8th frame
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    # Combine results to find the best matching frame
    best_frame = None
    best_score = 0
    best_time = None
    for i, frame in enumerate(frames):
        frame_result = analyze_frame(frame, i, keywords, tokenizer, model)

        score = frame_result[1]
        if score > best_score:
            best_score = score
            # Get the duration of the video
            duration = cap.get(cv2.CAP_PROP_POS_MSEC)

            # Get the total number of frames in the video
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Calculate the time per frame
            time_per_frame = duration / total_frames

            best_time = i * time_per_frame
            best_result = frame_result[0]



    # Get the timeframe
    start_time = max(0, (best_time - 10000))  # 5 seconds before the best frame
    end_time = min(cap.get(cv2.CAP_PROP_POS_MSEC), (best_time + 10000))  # 5 seconds after the best frame

    # Cut the downloaded video into the relevant timeframe and rename it based on the phrase input
    video_title = phrase.replace(' ', '_') + '.mp4'
    output_path = os.path.join('downloads', phrase + '.mp4')
    print(output_path)
    index_phrase = phrases.index(phrase)

    os.system(f'ffmpeg -loglevel quiet -ss {start_time/1000} -i {os.path.join("downloads", video_title)} -t {(end_time-start_time)/1000} -c copy "{output_path}"')

    print(f"Downloaded clip for phrase '{phrase}'")
    os.remove(os.path.join("downloads", video_title))

print("Finished downloading video clips.")

