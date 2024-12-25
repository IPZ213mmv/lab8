import numpy as np
import tensorflow as tf

# 1. Генерація синтетичних даних
np.random.seed(42)  # Фіксований seed для відтворюваності
n_samples = 1000  # Кількість точок

# Вхідні дані X в інтервалі [0, 1]
X_data = np.random.uniform(0, 1, n_samples).reshape(-1, 1)
# Вихідні дані y: формула 2x + 3 + шум
noise = np.random.normal(0, 0.1, n_samples).reshape(-1, 1)
y_data = 2 * X_data + 3 + noise

# 2. Модель лінійної регресії на TensorFlow
# Очищення старих графів
tf.compat.v1.reset_default_graph()

# Створення placeholders для вхідних даних
X = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='X')
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='y')

# Параметри моделі: ваги (W) і зміщення (b)
W = tf.Variable(tf.random.normal([1, 1]), name='weights')
b = tf.Variable(tf.random.normal([1]), name='bias')

# Лінійна модель: y_pred = W * X + b
y_pred = tf.add(tf.matmul(X, W), b)

# Функція втрат: середньоквадратична помилка
loss = tf.reduce_mean(tf.square(y_pred - y))

# Оптимізатор: градієнтний спуск
learning_rate = 0.01
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# 3. Навчання моделі
n_epochs = 2000  # Кількість епох
batch_size = 100  # Розмір пакета
n_batches = n_samples // batch_size  # Кількість пакетів

# Запуск сесії TensorFlow
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(n_epochs):
        for i in range(n_batches):
            # Вибірка міні-пакета
            batch_start = i * batch_size
            batch_end = batch_start + batch_size
            X_batch = X_data[batch_start:batch_end]
            y_batch = y_data[batch_start:batch_end]

            # Оптимізація моделі
            _, current_loss = sess.run([optimizer, loss], feed_dict={X: X_batch, y: y_batch})

        # Логування результатів кожні 500 епох
        if (epoch + 1) % 500 == 0:
            print(f"Епоха {epoch + 1}, Loss: {current_loss:.4f}")

    # Підсумкові значення W і b
    final_W, final_b = sess.run([W, b])
    print(f"Фінальні параметри моделі: W = {final_W[0][0]:.4f}, b = {final_b[0]:.4f}")

# 4. Результати
print("Результати навчання завершено!")
