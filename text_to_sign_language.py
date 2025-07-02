import re
import os
import cv2
from collections import Counter
from Levenshtein import distance as levenshtein_distance
from sentence_transformers import SentenceTransformer, util

# Define the dataset path where videos are stored
dataset_path = r"D:\Nipu(imp)\e\datasets"

# Assamese stopwords
assamese_stopwords = {
    "কিন্তু", "আছে", "এটা", "এটি", "এইটো", "তাত", "নেকি", "কৰি", "লাগে", "তেওঁ",
    "তুমি", "আমি", "মই", "আপুনি", "নাই", "তোমাক", "যি", "কোন", "তেওঁলোকে",
    "কৰা", "কৰিছা", "কৰিছে", "পাৰো", "পাৰিব", "ক'ৰ", "আহিছে", "দিয়া", "গৈ", "যাম"
}

# Dataset of Assamese sentences
dataset_sentences = [
    "অনুগ্ৰহ কৰি আপুনি এইটো পুনৰাবৃত্তি কৰিব পাৰিব নেকি",
    "অনুগ্ৰহ কৰি লাহে লাহে কথা পাতিব পাৰিব নেকি",
    "আজি আপুনি আজৰি আছে নিকি",
    "আপুনি ইমান দয়ালু",
    "আপুনি কি কৰিছে",
    "আপুনি কি কৰে",
    "আপুনি কি বিচাৰে",
    "আপুনি কি ভাবিছে",
    "আপুনি কিয় কান্দিছে",
    "আপুনি কিয় খং কৰিছে",
    "আপুনি কোন",
    "আপুনি কোনখন কলেজ_স্কুলৰ হয়",
    "আপুনি কৰিব পাৰিব",
    "আপুনি ক’ৰ পৰা আহিছে",
    "আপুনি খালেনে",
    "আপোনাক কেনেকৈ সহায় কৰিম",
    "আপোনাৰ কেৰিয়াৰৰ বাবে কি পৰিকল্পনা কৰিছে",
    "আপোনাৰ ফোন নম্বৰটো কি",
    "আপোনাৰ বয়স কিমান",
    "আমি বাহিৰলৈ যাম নেকি",
    "কি কলা তাক",
    "কি কৰিম মই দোমোজাত পৰিছো",
    "কি হ'ব বিচাৰিছা",
    "কি হ’ল",
    "কিবা লাগে নেকি",
    "কিবা লুকুৱাইছা নেকি",
    "কেতিয়া যাব ৰেলখন",
    "কেনেকৈ বিশ্বাস কৰিম তোমাক",
    "কেনেকৈ সাহস কৰিলে",
    "চিন্তা কৰাৰ প্ৰয়োজন নাই চিন্তা নকৰিব",
    "তাত মই আপোনাক সহায় কৰিব নোৱাৰো",
    "তুমি কিয় হতাশ হৈছা",
    "তুমি যিয়ে নকৰা কিয়, মই একো গুৰুত্ব নিদিওঁ।",
    "তেওঁ _তাই মোৰ বন্ধু",
    "তোমাৰ লগত কথা পাতি ভাল লাগিল",
    "নমস্কাৰ",
    "নমস্কাৰ, আপোনাৰ কি খবৰ",
    "মই আপোনাক সহায় কৰিব পাৰো নে",
    "মই একমত নহয়",
    "মই ঠিকেই আছো। ধন্যবাদ ছাৰ",
    "মোক কোনোবাই ৰখাই দিলে",
    "মোৰ তোমাক ভাল লাগে_মই তোমাক ভাল পাওঁ",
    "মোৰ বাবে ইয়াৰ কোনো পাৰ্থক্য নাই",
    "যোৱা আৰু শুই দিয়া",
    "সঁচা কথা কোৱা",
    "সি কোঠাটোত সোমাই গৈছে",
    "সি গৈ আছে"
]

# Load sentence transformer model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Preprocessing: clean text
def preprocess(sentence):
    sentence = re.sub(r'[^ঀ-৿\s]', '', sentence)
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    return sentence

# Glossing: tokenize + remove stopwords
def gloss(sentence):
    tokens = preprocess(sentence).split()
    return [token for token in tokens if token not in assamese_stopwords]

# Get video path
def get_video_path(sentence):
    return os.path.join(dataset_path, sentence, "ই.mp4")

# Play video
def play_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Sign Language Video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# --- Single User Input ---
print("Welcome to the Assamese Text to Sign Language System.")
input_sentence = input("Enter an Assamese sentence: ").strip()

if not input_sentence:
    print("No input provided. Exiting.")
else:
    # Try sentence-transformer based matching
    input_vec = model.encode(input_sentence, convert_to_tensor=True)
    dataset_vecs = model.encode(dataset_sentences, convert_to_tensor=True)
    sim_scores = util.cos_sim(input_vec, dataset_vecs)[0]
    best_idx = sim_scores.argmax().item()
    best_match = dataset_sentences[best_idx]
    print(f"Matched by sentence embedding: {best_match} (Score: {sim_scores[best_idx].item():.2f})")

    # Play corresponding video
    video_path = get_video_path(best_match)
    if os.path.exists(video_path):
        play_video(video_path)
    else:
        print(f"Video file not found at: {video_path}")
