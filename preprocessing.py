import os
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm

def get_features(path, sample_rate=22050):
    try:
        audio, sr = librosa.load(path, sr=sample_rate, mono=True)
        # Adjust n_fft dynamically if the audio is too short
        n_fft = min(1024, len(audio)//2)  # Use len(audio)//2 or 1024, whichever is smaller
        # Extract features
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40, n_fft=n_fft).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=n_fft).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=n_fft).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr).T, axis=0)
        
        return np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return np.zeros((193,))  # Return a placeholder feature vector if an error occurs

# Load RAVDESS dataset
def load_ravdess_dataset(ravdess_path):
    files = []
    emotion_map = {
        1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry',
        6: 'fearful', 7: 'disgust', 8: 'surprised'
    }

    for dirpath, _, filenames in os.walk(ravdess_path):
        for file in filenames:
            if file.endswith(".wav"):
                parts = file.split('-')


                if len(parts) >= 7:  # Ensure the filename has enough parts
                    emotion_code = int(parts[2])  # Example: "06" (fearful)
                    actor_code_str = parts[6].replace('.wav', '')  # Remove '.wav' and use the actor code
                    actor_code = int(actor_code_str)  # Example: "12" (female)

                    # Map emotion code to emotion label
                    emotion = emotion_map.get(emotion_code, 'unknown')

                    # Gender is based on whether the actor number is odd (male) or even (female)
                    gender = 'male' if actor_code % 2 != 0 else 'female'

                    files.append((os.path.join(dirpath, file), emotion, gender))
                else:
                    print(f"Skipping file {file} due to unexpected format.")
    
    df = pd.DataFrame(files, columns=['Path', 'Emotion', 'Gender'])
    return df

def load_crema_dataset(crema_path):
    files = []
    crema_gender_map = {
        # Example: Map actor IDs to gender (add all IDs here)
        '1001': 'male', '1002': 'female', '1003': 'male', '1004': 'female'
    }
    emotion_map = {
        'SAD': 'sad', 'ANG': 'angry', 'DIS': 'disgust',
        'FEA': 'fearful', 'HAP': 'happy', 'NEU': 'neutral'
    }
    
    for file in os.listdir(crema_path):
        if file.endswith(".wav"):
            parts = file.split('_')
            
            # Ensure the filename has at least 3 parts to correctly identify emotion
            if len(parts) >= 3:
                emotion_code = parts[2]  # Emotion code should be the third part
                actor_id = parts[0]  # Actor ID is the first part
                
                
                # Map emotion code to emotion label, if valid
                emotion = emotion_map.get(emotion_code, 'unknown')
                gender = crema_gender_map.get(actor_id, 'unknown')
                
                # Add the file info to the list
                files.append((os.path.join(crema_path, file), emotion, gender))
    
    df = pd.DataFrame(files, columns=['Path', 'Emotion', 'Gender'])
    return df

# Main preprocessing pipeline
def preprocess_datasets(ravdess_path, crema_path, output_features_path):
    # Load datasets
    ravdess_df = load_ravdess_dataset(ravdess_path)
    crema_df = load_crema_dataset(crema_path)

    # Combine datasets
    combined_df = pd.concat([ravdess_df, crema_df], axis=0)
    combined_df.reset_index(drop=True, inplace=True)

    # Print unique values of Emotion and Gender to debug
    print("Unique emotions:", combined_df['Emotion'].unique())
    print("Unique genders:", combined_df['Gender'].unique())

    # Extract features
    X, Y, G = [], [], []
    for path, emotion, gender in tqdm(zip(combined_df['Path'], combined_df['Emotion'], combined_df['Gender']),
                                      total=len(combined_df)):
        features = get_features(path)
        X.append(features)
        Y.append(emotion)
        G.append(gender)

    # Create DataFrame for features
    feature_df = pd.DataFrame(X)
    feature_df['Emotion'] = Y
    feature_df['Gender'] = G

    # Save to CSV
    feature_df.to_csv(output_features_path, index=False)
    print(f"Features saved to {output_features_path}")

# Paths to datasets
ravdess_path = "Updated/ravdess"  # Replace with your RAVDESS dataset path
crema_path = "Updated/cremad"  # Replace with your CREMA-D dataset path
output_features_path = "features_combined.csv"

# Run preprocessing
preprocess_datasets(ravdess_path, crema_path, output_features_path)
