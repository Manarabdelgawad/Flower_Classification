import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

data_path = "D:\\classification_flowers\\data\\flowers"          
output = "D:\\classification_flowers\\outputs"
model_path = "best_model.h5"            
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20


# Split dataset
def split_dataset(source_dir, output_dir, val_size=0.15, test_size=0.15, random_state=42):
    os.makedirs(output_dir, exist_ok=True)
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        train_files, temp = train_test_split(files, test_size=val_size+test_size, random_state=random_state)
        val_files, test_files = train_test_split(temp, test_size=0.5, random_state=random_state)
        
        for split_name, split_files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            dest = os.path.join(output_dir, split_name, cls)
            os.makedirs(dest, exist_ok=True)
            for f in split_files:
                shutil.copy(os.path.join(cls_path, f), dest)

split_dataset(data_path, output)

def create_dataframe(split_dir):
    data = []
    for cls in os.listdir(split_dir):
        cls_path = os.path.join(split_dir, cls)
        for img in os.listdir(cls_path):
            data.append({
                'filename': os.path.join(cls_path, img),
                'class': cls
            })
    return pd.DataFrame(data)

train_df = create_dataframe(os.path.join(output, 'train'))
val_df   = create_dataframe(os.path.join(output, 'val'))
test_df  = create_dataframe(os.path.join(output, 'test'))

#  Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    width_shift_range=0.2, 
    height_shift_range=0.2
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df, x_col="filename", y_col="class",
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)

val_generator = val_test_datagen.flow_from_dataframe(
    val_df, x_col="filename", y_col="class",
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)

test_generator = val_test_datagen.flow_from_dataframe(
    test_df, x_col="filename", y_col="class",
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False
)

num_classes = len(train_generator.class_indices)
print(f"Number of classes: {num_classes}")

#  model architecture
input_tensor = Input(shape=(224, 224, 3))

base_model = ResNet50(
    include_top=False, 
    weights='imagenet', 
    input_tensor=input_tensor
)
base_model.trainable = False

x = base_model.output
x = Dropout(0.4)(x)
x = BatchNormalization()(x)
x = Dense(128, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.5)(x)

predictions = Dense(num_classes, activation='softmax')(x)  

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=0.0001), 
    loss="categorical_crossentropy", 
    metrics=["accuracy"]
)

# Print model summary
model.summary()

# Train
callbacks = [ModelCheckpoint(model_path, save_best_only=True, verbose=1)]

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Plot training curves
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Training Curves')
plt.savefig(os.path.join(output, "training_curves.png"))
plt.show()

print("training completed")