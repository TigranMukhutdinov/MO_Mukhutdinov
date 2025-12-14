
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import numpy as np

#Загрузка данных
print("Загрузка arrows.npz...")


with np.load('arrows.npz', allow_pickle=True) as data:

    arrays = []
    for key in data.files:
        arrays.append(data[key])

    if len(arrays) >= 4:
        x_train = arrays[0]
        y_train = arrays[1]
        x_test = arrays[4]
        y_test = arrays[5]


    else:
        print(f"Ошибка: в файле только {len(arrays)} массива(ов), нужно минимум 4")
        exit()

#Информация о датасете
print(f"\nИнформация о датасете:")
print(f"x_train: {x_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"x_test: {x_test.shape}")
print(f"y_test: {y_test.shape}")
print(f"Классы: {np.unique(y_train)}")
print(f"Всего классов: {len(np.unique(y_train))}")


# Нормализация
if x_train.max() > 1.0:
    x_train, x_test = x_train / 255.0, x_test / 255.0

# Добавление channels dimension
if len(x_train.shape) == 3:
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

#Создание датасета
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


#Модель
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Для маленьких 8x8 изображений
        self.conv1 = Conv2D(16, 3, activation='relu', padding='same')
        self.flatten = Flatten()
        # Количество нейронов уменьшено для маленького датасета
        self.d1 = Dense(64, activation='relu')
        # Выходной слой: столько нейронов, сколько классов
        n_classes = len(np.unique(y_train))
        self.d2 = Dense(n_classes)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


# Создание модели
model = MyModel()

#Оптимизатор и функция потерь
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

#Метрики
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


#Функция обучения
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)


#Цикл обучения
EPOCHS = 5

print("\n" + "=" * 50)
print("НАЧАЛО ОБУЧЕНИЯ")
print("=" * 50)

for epoch in range(EPOCHS):
    # Сброс метрик
    train_loss.reset_state()
    train_accuracy.reset_state()
    test_loss.reset_state()
    test_accuracy.reset_state()

    # Обучение
    for images, labels in train_ds:
        train_step(images, labels)

    # Тестирование
    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    # Вывод результатов
    print(f'Эпоха {epoch + 1}, '
          f'Loss: {train_loss.result():.4f}, '
          f'Accuracy: {train_accuracy.result() * 100:.2f}, '
          f'Test Loss: {test_loss.result():.4f}, '
          f'Test Accuracy: {test_accuracy.result() * 100:.2f}')

print("\n" + "=" * 50)
print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
print("=" * 50)

