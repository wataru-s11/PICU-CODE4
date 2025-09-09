import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers

# 画像サイズとパス設定
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 10
TRAIN_DIR = r"C:\Users\sakai\OneDrive\Desktop\BOT\CVP"

# 前処理 + Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=3,
    zoom_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05,
    brightness_range=(0.7, 1.3),
    shear_range=0.05,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# クラス数とインデックス保存
num_classes = len(train_generator.class_indices)
print("クラスインデックス:", train_generator.class_indices)
with open("class_indices.json", "w", encoding="utf-8") as f:
    json.dump(train_generator.class_indices, f, ensure_ascii=False, indent=2)

# モデル構築（VGG風）
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 学習
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# モデル保存（.keras 推奨）
model.save("cvp_model.keras")
print("✅ モデル保存完了")

# 損失と精度のプロット保存（任意）
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.title('Accuracy')
plt.savefig("cvp_accuracy.png")
plt.clf()

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.title('Loss')
plt.savefig("cvp_loss.png")
plt.clf()
