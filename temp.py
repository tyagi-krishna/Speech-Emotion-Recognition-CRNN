import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Custom layer needed for loading the model
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                shape=(input_shape[-1], 1),
                                initializer='random_normal',
                                trainable=True)
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, inputs):
        score = tf.matmul(inputs, self.W)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = inputs * attention_weights
        return tf.reduce_sum(context_vector, axis=1)

# Load and preprocess test data
def prepare_data(data_path):
    # Load the dataset
    df = pd.read_csv(data_path)
    df = df[df['Gender'] != 'Unknown']
    
    # Extract features and labels
    X = df.drop(columns=['Emotion', 'Gender']).values
    y_emotion = df['Emotion'].values
    y_gender = df['Gender'].values
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Encode labels
    label_encoder_emotion = LabelEncoder()
    y_emotion_encoded = label_encoder_emotion.fit_transform(y_emotion)
    y_emotion_onehot = tf.keras.utils.to_categorical(y_emotion_encoded)
    
    label_encoder_gender = LabelEncoder()
    y_gender_encoded = label_encoder_gender.fit_transform(y_gender)
    y_gender_onehot = tf.keras.utils.to_categorical(y_gender_encoded)
    
    # Reshape for CRNN
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return (X, y_emotion_onehot, y_gender_onehot, 
            label_encoder_emotion.classes_, label_encoder_gender.classes_)

def evaluate_model(model, X, y_emotion, y_gender, emotion_classes, gender_classes):
    # Make predictions
    y_pred = model.predict(X)
    y_pred_emotion = np.argmax(y_pred[0], axis=1)
    y_pred_gender = np.argmax(y_pred[1], axis=1)
    
    y_true_emotion = np.argmax(y_emotion, axis=1)
    y_true_gender = np.argmax(y_gender, axis=1)
    
    # Calculate and print accuracies
    emotion_accuracy = np.mean(y_pred_emotion == y_true_emotion)
    gender_accuracy = np.mean(y_pred_gender == y_true_gender)
    
    print(f"\nEmotion Classification Accuracy: {emotion_accuracy * 100:.2f}%")
    print(f"Gender Classification Accuracy: {gender_accuracy * 100:.2f}%")
    
    # Print detailed classification reports
    print("\nEmotion Classification Report:")
    print(classification_report(y_true_emotion, y_pred_emotion,
                              target_names=emotion_classes))
    
    print("\nGender Classification Report:")
    print(classification_report(y_true_gender, y_pred_gender,
                              target_names=gender_classes))

def main():
    # File paths
    model_path = "improved_crnn_model_gender_emotion.keras"
    data_path = "features_combined.csv"  # Use your data file path
    
    try:
        # Load the trained model
        print("Loading model...")
        model = load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})
        print("Model loaded successfully!")
        
        # Prepare the test data
        print("Preparing data...")
        X, y_emotion, y_gender, emotion_classes, gender_classes = prepare_data(data_path)
        print("Data prepared successfully!")
        
        # Evaluate the model
        print("Evaluating model...")
        evaluate_model(model, X, y_emotion, y_gender, emotion_classes, gender_classes)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()