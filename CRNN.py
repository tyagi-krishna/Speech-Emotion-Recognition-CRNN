import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, Input, BatchNormalization, Bidirectional
from tensorflow.keras.layers import GlobalAveragePooling1D, Concatenate, Add, LayerNormalization, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load and preprocess the dataset
df = pd.read_csv('features_combined.csv')
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
y_emotion_onehot = to_categorical(y_emotion_encoded)

label_encoder_gender = LabelEncoder()
y_gender_encoded = label_encoder_gender.fit_transform(y_gender)
y_gender_onehot = to_categorical(y_gender_encoded)

# Combine the emotion and gender labels into a tuple for stratification
combined_labels = list(zip(y_emotion_encoded, y_gender_encoded))

# Perform the train-test split with stratification based on combined labels
X_train, X_test, y_train_emotion, y_test_emotion, y_train_gender, y_test_gender = train_test_split(
    X, y_emotion_onehot, y_gender_onehot, test_size=0.2, random_state=42,
    stratify=combined_labels  # Stratify based on the combined labels
)


# Reshape for CRNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

# Attention Layer Definition
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

# Model Building
def build_improved_crnn(input_shape, num_classes_emotion, num_classes_gender):
    inputs = Input(shape=input_shape)
    
    # First CNN block
    x1 = Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x1 = LayerNormalization()(x1)
    x1 = Conv1D(64, 3, activation='relu', padding='same')(x1)
    x1 = LayerNormalization()(x1)
    x1 = MaxPooling1D(2)(x1)
    x1 = Dropout(0.3)(x1)
    
    # Second CNN block
    x2 = Conv1D(128, 3, activation='relu', padding='same')(x1)
    x2 = LayerNormalization()(x2)
    x2 = Conv1D(128, 3, activation='relu', padding='same')(x2)
    x2 = LayerNormalization()(x2)
    x2 = MaxPooling1D(2)(x2)
    x2 = Dropout(0.3)(x2)
    
    # Third CNN block with residual connection
    x3 = Conv1D(256, 3, activation='relu', padding='same')(x2)
    x3 = LayerNormalization()(x3)
    x3 = Conv1D(256, 3, activation='relu', padding='same')(x3)
    x3 = LayerNormalization()(x3)
    x3_pool = MaxPooling1D(2)(x3)
    x3_res = Dropout(0.3)(x3_pool)
    
    # Parallel LSTM paths
    lstm1 = Bidirectional(LSTM(128, return_sequences=True))(x3_res)
    lstm1 = Dropout(0.3)(lstm1)
    
    lstm2 = Bidirectional(LSTM(128, return_sequences=True))(x3_res)
    lstm2 = Dropout(0.3)(lstm2)
    
    # Apply attention mechanism
    att1 = AttentionLayer()(lstm1)
    att2 = AttentionLayer()(lstm2)
    
    # Combine features
    combined = Concatenate()([att1, att2])
    
    # Dense layers for emotion classification (deeper path)
    emotion_dense = Dense(256, activation='relu')(combined)
    emotion_dense = Dropout(0.4)(emotion_dense)
    emotion_dense = Dense(128, activation='relu')(emotion_dense)
    emotion_dense = Dropout(0.4)(emotion_dense)
    emotion_output = Dense(num_classes_emotion, activation='softmax', name="Emotion")(emotion_dense)
    
    # Dense layers for gender classification
    gender_dense = Dense(128, activation='relu')(combined)
    gender_dense = Dropout(0.3)(gender_dense)
    gender_output = Dense(num_classes_gender, activation='softmax', name="Gender")(gender_dense)
    
    model = Model(inputs, [emotion_output, gender_output])
    return model

# Build and compile model
input_shape = X_train.shape[1:]
num_classes_emotion = y_emotion_onehot.shape[1]
num_classes_gender = y_gender_onehot.shape[1]

model = build_improved_crnn(input_shape, num_classes_emotion, num_classes_gender)

# Compile with fixed learning rate and loss weights
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={
        'Emotion': 'categorical_crossentropy',
        'Gender': 'categorical_crossentropy'
    },
    loss_weights={
        'Emotion': 1.0,
        'Gender': 0.5
    },
    metrics={'Emotion': 'accuracy', 'Gender': 'accuracy'}
)

# Adjusted callbacks with longer patience
early_stopping = EarlyStopping(
    monitor='val_Emotion_accuracy',
    patience=20,  # Increased patience
    restore_best_weights=True,
    mode='max'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_Emotion_accuracy',
    factor=0.2,  # Smaller reduction factor
    patience=10,  # Increased patience
    min_lr=1e-6,
    mode='max',
    verbose=1
)

# Train with adjusted parameters
history = model.fit(
    X_train,
    {'Emotion': y_train_emotion, 'Gender': y_train_gender},
    validation_data=(X_test, {'Emotion': y_test_emotion, 'Gender': y_test_gender}),
    epochs=150,  # Increased maximum epochs
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=2
)

# Save the model
model.save("improved_crnn_model_gender_emotion.keras")