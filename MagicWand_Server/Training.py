import tensorflow as tf
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split

# =============================
# 配置参数
# =============================
IMAGE_SIZE = 24
DATA_DIR = 'dataset'  # 数据集根目录
BATCH_SIZE = 32
EPOCHS = 100
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42
MODEL_NAME = 'shape_classifier'

# GPU 设置
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[INFO] 使用 {len(physical_devices)} 个 GPU")
    except RuntimeError as e:
        print(e)


# =============================
# 1. 加载所有类别的图像数据
# =============================
def load_bitmaps_from_file(filepath):
    """从 number.txt 读取所有 24x24 的 01 图像"""
    images = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip() and len(line.strip()) == IMAGE_SIZE]

        # 检查是否能被24整除
        if len(lines) < IMAGE_SIZE:
            print(f"[警告] {filepath} 中有效行数不足 24 行，跳过。")
            return []

        num_images = len(lines) // IMAGE_SIZE
        for i in range(num_images):
            block = lines[i * IMAGE_SIZE: (i + 1) * IMAGE_SIZE]
            if len(block) != IMAGE_SIZE:
                continue
            try:
                img = np.array([[1.0 if c == '1' else 0.0 for c in line] for line in block])
                images.append(img)
            except Exception as e:
                print(f"[错误] 解析图像块 {i} 失败：{e}")
                continue
    except Exception as e:
        print(f"[错误] 无法读取文件 {filepath}: {e}")
        return []

    print(f"[加载] 从 {os.path.basename(os.path.dirname(filepath))} 加载 {len(images)} 张图像")
    return images


# 自动发现所有类别（子文件夹）
class_names = sorted([
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
])

if not class_names:
    raise FileNotFoundError(f"未在 {DATA_DIR} 找到任何类别文件夹！")

print(f"[INFO] 发现 {len(class_names)} 个类别: {class_names}")
label_to_id = {name: idx for idx, name in enumerate(class_names)}

# 收集所有图像和标签
all_images = []
all_labels = []

for class_name in class_names:
    folder_path = os.path.join(DATA_DIR, class_name)
    file_path = os.path.join(folder_path, 'number.txt')

    if not os.path.exists(file_path):
        print(f"[警告] 缺少文件: {file_path}")
        continue

    images = load_bitmaps_from_file(file_path)
    labels = [label_to_id[class_name]] * len(images)

    all_images.extend(images)
    all_labels.extend(labels)

# 转为 numpy 数组
X = np.array(all_images, dtype=np.float32)  # 形状: (N, 24, 24)
y = np.array(all_labels, dtype=np.int32)

print(f"[INFO] 总样本数: {len(X)}")
print(f"        输入形状: {X.shape}, 标签形状: {y.shape}")

if len(X) == 0:
    raise ValueError("没有加载到任何有效图像，请检查数据格式！")

# =============================
# 2. 划分训练/验证集
# =============================
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=VALIDATION_SPLIT,
    stratify=y,
    random_state=RANDOM_SEED
)

# 增加通道维度 (H, W) -> (H, W, 1)
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]

# 构建 tf.data 数据集
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))

train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# =============================
# 3. 构建 CNN 模型
# =============================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),

    # 自动归一化（均值为0，方差为1）
    tf.keras.layers.Normalization(axis=-1),

    # 卷积块 1
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),  # 24 -> 12
    tf.keras.layers.Dropout(0.25),

    # 卷积块 2
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),  # 12 -> 6
    tf.keras.layers.Dropout(0.25),

    # 卷积块 3（可选）
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),  # 6 -> 3
    tf.keras.layers.Dropout(0.25),

    # 全连接层
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(class_names), activation='softmax', name='predictions')
])

# 适配 Normalization 层（计算均值和标准差）
norm_layer = model.layers[1]
norm_layer.adapt(train_ds.map(lambda x, y: x))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =============================
# 4. 训练模型
# =============================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

print("\n[INFO] 开始训练...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# =============================
# 5. 评估模型
# =============================
print("\n[RESULT] 验证集性能:")
results = model.evaluate(val_ds, verbose=0)
print(f"  损失: {results[0]:.4f}, 准确率: {results[1]:.4f}")

# =============================
# 6. 保存模型
# =============================
# 保存为 Keras 模型
model.save(f'{MODEL_NAME}.keras')
print(f"\n✅ 模型已保存为: {MODEL_NAME}.keras")

# 转换为 TFLite（适用于嵌入式设备如 ESP32）
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# 可选：量化以减小体积（推荐用于部署）
# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
tflite_model = converter.convert()

with open(f'{MODEL_NAME}.tflite', 'wb') as f:
    f.write(tflite_model)
print(f"✅ TFLite 模型已保存为: {MODEL_NAME}.tflite")


# =============================
# 7. 测试单张图像预测（可选）
# =============================
def predict_from_file(model, file_path, class_names):
    """从一个文本文件中读取第一张 24x24 图像并预测"""
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()][:24]
    if len(lines) != 24:
        raise ValueError("图像必须正好 24 行")
    if any(len(line) != 24 for line in lines):
        raise ValueError("每行必须正好 24 列")

    # 转为 0/1 数值矩阵
    img = np.array([[1.0 if c == '1' else 0.0 for c in line] for line in lines], dtype=np.float32)
    img = img[None, ..., None]  # 添加 batch 和 channel 维度

    pred = model.predict(img, verbose=0)
    class_id = np.argmax(pred[0])
    confidence = np.max(pred[0])

    print(f"预测类别: {class_names[class_id]} (置信度: {confidence:.4f})")
    return class_id, confidence

# 示例：预测某个图形（取消注释使用）
predict_from_file(model, 'dataset/circle/number.txt', class_names)