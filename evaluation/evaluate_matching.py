import torchaudio
import torch
import os
import re
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from Levenshtein import distance as levenshtein_distance
from jiwer import wer
import pandas as pd

# ----------- Setup -------------
SAMPLE_RATE = 16000
MODEL_ID = "manandey/wav2vec2-large-xlsr-assamese"
AUDIO_DIR = r"D:\Nipu(imp)\e\audio_samples"
EXPECTED_MATCHES = r"D:\Nipu(imp)\e\evaluation\expected_matches.txt"
OUTPUT_CSV = r"D:\Nipu(imp)\e\evaluation\evaluation_results_speakers_1_2.csv"

# Load model
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

# Assamese stopwords and dataset (47 sentences)
assamese_stopwords = {
    "কিন্তু", "আছে", "এটা", "এটি", "এইটো", "তাত", "নেকি", "কৰি", "লাগে", "তেওঁ",
    "তুমি", "আমি", "মই", "আপুনি", "নাই", "তোমাক", "যি", "কোন", "তেওঁলোকে",
    "কৰা", "কৰিছা", "কৰিছে", "পাৰো", "পাৰিব", "ক'ৰ", "আহিছে", "দিয়া", "গৈ", "যাম"
}

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
    "আপুনি কিয় খং কৰি�ছে",
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

# Load ground truth for Speakers 1 and 2 (94 lines: 47 sentences x 2 speakers)
def load_ground_truth(file_path):
    ground_truth = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    
    if len(sentences) != 141:
        print(f"Warning: Expected 141 sentences, found {len(sentences)}")
        return None
    
    # Extract sentences for Speakers 1 and 2 (first and second line per sentence)
    for sentence_idx, i in enumerate(range(0, len(sentences), 3), start=1):
        # Speaker 1: first line (index i)
        ground_truth[f"{sentence_idx}_1.wav"] = sentences[i]
        # Speaker 2: second line (index i+1)
        if i + 1 < len(sentences):
            ground_truth[f"{sentence_idx}_2.wav"] = sentences[i + 1]
    
    return ground_truth

# ----------- Functions -------------
def load_wav_file(file_path):
    waveform, sr = torchaudio.load(file_path)
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)
    return waveform.squeeze().numpy()

def transcribe(audio):
    input_values = processor(audio, return_tensors="pt", sampling_rate=SAMPLE_RATE).input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription.strip()

def preprocess(text):
    text = re.sub(r'[^\u0980-\u09FF\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def gloss(text):
    tokens = preprocess(text).split()
    return [t for t in tokens if t not in assamese_stopwords]

def jaccard_similarity(tokens1, tokens2):
    s1, s2 = set(tokens1), set(tokens2)
    if not s1 or not s2:
        return 0
    return len(s1 & s2) / len(s1 | s2)

def find_best_match(input_text):
    input_gloss = gloss(input_text)
    glossed_dataset = [gloss(sent) for sent in dataset_sentences]
    best_score, best_sentence = 0.0, None

    for sent, glossed in zip(dataset_sentences, glossed_dataset):
        score = jaccard_similarity(input_gloss, glossed)
        if score > best_score:
            best_score, best_sentence = score, sent

    if best_score >= 0.2:
        return best_sentence, best_score

    distances = [(s, levenshtein_distance(preprocess(input_text), preprocess(s))) for s in dataset_sentences]
    best_sentence, dist = min(distances, key=lambda x: x[1])
    return best_sentence, 1 - dist / max(len(input_text), 1)

def evaluate_wav_files(audio_dir, ground_truth):
    results = []
    for file_name in os.listdir(audio_dir):
        if file_name.endswith(".wav") and file_name in ground_truth:
            # Only process files for Speakers 1 and 2
            speaker_id = file_name.split('_')[1].replace('.wav', '')
            if speaker_id in ['1', '2']:
                file_path = os.path.join(audio_dir, file_name)
                audio = load_wav_file(file_path)
                transcription = transcribe(audio)
                best_sentence, score = find_best_match(transcription)
                
                gt_sentence = ground_truth[file_name]
                transcription_wer = wer(preprocess(gt_sentence), preprocess(transcription))
                is_correct_match = best_sentence == gt_sentence
                
                results.append({
                    "File": file_name,
                    "Ground Truth": gt_sentence,
                    "Transcription": transcription,
                    "WER": transcription_wer,
                    "Matched Sentence": best_sentence,
                    "Jaccard Score": score,
                    "Correct Match": is_correct_match
                })
    
    return pd.DataFrame(results)

# ----------- Main Evaluation -------------
def main():
    if not os.path.exists(AUDIO_DIR):
        print(f"Error: Audio directory {AUDIO_DIR} not found.")
        return
    if not os.path.exists(EXPECTED_MATCHES):
        print(f"Error: Ground truth file {EXPECTED_MATCHES} not found.")
        return

    print("Loading ground truth for Speakers 1 and 2...")
    ground_truth = load_ground_truth(EXPECTED_MATCHES)
    if ground_truth is None:
        return

    print("Evaluating WAV files for Speakers 1 and 2...")
    results_df = evaluate_wav_files(AUDIO_DIR, ground_truth)
    
    if results_df.empty:
        print("No valid WAV files found for Speakers 1 and 2.")
        return
    
    # Summary
    print("\n=== Summary ===")
    print(f"Total Files Evaluated: {len(results_df)}")
    print(f"Average WER: {results_df['WER'].mean():.4f}")
    print(f"Sentence Matching Accuracy: {results_df['Correct Match'].mean():.4f}")
    
    # Per-speaker performance
    results_df['Speaker'] = results_df['File'].str.split('_').str[1].str.replace('.wav', '')
    print("\n=== Per-Speaker Performance ===")
    speaker_summary = results_df.groupby('Speaker').agg({
        'WER': 'mean',
        'Correct Match': 'mean',
        'File': 'count'
    }).rename(columns={'File': 'Count'})
    print(speaker_summary.to_string())
    
    # Save results
    results_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"\nResults saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()