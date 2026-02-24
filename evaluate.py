import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR = "D:\\classification_flowers\\data\\flowers"

model = load_model("models/best_model.h5")

datagen = ImageDataGenerator(rescale=1./255)

generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

preds = model.predict(generator)
y_pred = np.argmax(preds, axis=1)
y_true = generator.classes

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=generator.class_indices.keys(),
            yticklabels=generator.class_indices.keys())
plt.savefig("outputs/confusion_matrix.png")

print(classification_report(y_true, y_pred))