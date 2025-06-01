import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import time

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# One-hot encode labels
y_train_one_hot = tf.keras.utils.to_categorical(y_train, 10)
y_test_one_hot = tf.keras.utils.to_categorical(y_test, 10)

# Split data
val_split = 0.2
split_idx = int(len(x_train) * (1 - val_split))
x_train_split = x_train[:split_idx]
x_val = x_train[split_idx:]
y_train_split = y_train_one_hot[:split_idx]
y_val = y_train_one_hot[split_idx:]

print(f"Training set: {x_train_split.shape}")
print(f"Validation set: {x_val.shape}")
print(f"Test set: {x_test.shape}")

# Enhanced Teacher model (to ensure it performs better)
def build_teacher_model():
    model = models.Sequential([
        layers.Conv2D(64, 3, activation='relu', padding='same', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),
        layers.BatchNormalization(),
        
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),
        layers.BatchNormalization(),
        
        layers.Conv2D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Very simple student model
def build_student_model():
    model = models.Sequential([
        layers.Conv2D(16, 3, activation='relu', padding='same', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# 1. Train teacher model with longer training to ensure high performance
print("Training teacher model...")
teacher = build_teacher_model()
teacher.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Use callbacks to ensure we get the best model
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
]

teacher_history = teacher.fit(
    x_train_split, y_train_split,
    validation_data=(x_val, y_val),
    epochs=20,  # More epochs for better performance
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)

# Evaluate teacher
teacher_loss, teacher_acc = teacher.evaluate(x_test, y_test_one_hot, verbose=0)
print(f"Teacher model accuracy: {teacher_acc*100:.2f}%")

# 2. Train a vanilla student (no distillation) for comparison
print("Training vanilla student model...")
vanilla_student = build_student_model()
vanilla_student.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

vanilla_history = vanilla_student.fit(
    x_train_split, y_train_split,
    validation_data=(x_val, y_val),
    epochs=10,
    batch_size=128,
    verbose=1
)

# Evaluate vanilla student
vanilla_loss, vanilla_acc = vanilla_student.evaluate(x_test, y_test_one_hot, verbose=0)
print(f"Vanilla student accuracy: {vanilla_acc*100:.2f}%")

# 3. Knowledge Distillation (explicit implementation)
print("Implementing knowledge distillation...")

# Temperature parameter
temperature = 5.0  # Higher temperature for more softening

# Alpha parameter (weight between hard and soft targets)
alpha = 0.3  # More weight on soft targets to benefit from teacher's knowledge

# Get teacher predictions on training and validation data
print("Generating soft targets from teacher...")
teacher_preds_train = teacher.predict(x_train_split, verbose=0)
teacher_preds_val = teacher.predict(x_val, verbose=0)

# Apply temperature scaling
def apply_temperature(probs, temperature):
    # Convert to logits
    logits = np.log(probs + 1e-10)
    # Apply temperature
    logits_t = logits / temperature
    # Convert back to probabilities
    probs_t = np.exp(logits_t) / np.sum(np.exp(logits_t), axis=1, keepdims=True)
    return probs_t

# Soften predictions
soft_targets_train = apply_temperature(teacher_preds_train, temperature)
soft_targets_val = apply_temperature(teacher_preds_val, temperature)

# Create a new student model for KD
kd_student = build_student_model()
kd_student.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Custom training loop for knowledge distillation
epochs = 15  # More epochs for better knowledge transfer
batch_size = 128
steps_per_epoch = len(x_train_split) // batch_size
validation_steps = max(1, len(x_val) // batch_size)

# Initialize history dictionary
kd_history = {
    'loss': [],
    'accuracy': [],
    'val_loss': [],
    'val_accuracy': []
}

print("Training student with knowledge distillation...")
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    
    # Training
    train_loss = 0
    train_acc = 0
    
    # Shuffle training data
    indices = np.random.permutation(len(x_train_split))
    x_shuffled = x_train_split[indices]
    y_hard_shuffled = y_train_split[indices]
    y_soft_shuffled = soft_targets_train[indices]
    
    for step in range(steps_per_epoch):
        # Get batch
        start_idx = step * batch_size
        end_idx = min(start_idx + batch_size, len(x_shuffled))
        x_batch = x_shuffled[start_idx:end_idx]
        y_hard_batch = y_hard_shuffled[start_idx:end_idx]
        y_soft_batch = y_soft_shuffled[start_idx:end_idx]
        
        # Forward pass
        with tf.GradientTape() as tape:
            predictions = kd_student(x_batch, training=True)
            
            # Hard target loss (cross-entropy with true labels)
            hard_loss = tf.keras.losses.categorical_crossentropy(y_hard_batch, predictions)
            
            # Soft target loss (KL divergence with teacher predictions)
            soft_loss = tf.keras.losses.kullback_leibler_divergence(y_soft_batch, predictions)
            
            # Combined loss with temperature scaling factor for soft loss
            loss = alpha * hard_loss + (1 - alpha) * soft_loss * (temperature ** 2)
            
        # Compute gradients and update weights
        gradients = tape.gradient(loss, kd_student.trainable_weights)
        kd_student.optimizer.apply_gradients(zip(gradients, kd_student.trainable_weights))
        
        # Update metrics
        train_loss += tf.reduce_mean(loss)
        train_acc += tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(y_hard_batch, axis=1), tf.argmax(predictions, axis=1)), 
            tf.float32))
    
    # Average training metrics
    train_loss /= steps_per_epoch
    train_acc /= steps_per_epoch
    
    # Validation
    val_loss = 0
    val_acc = 0
    
    for step in range(validation_steps):
        # Get batch
        start_idx = step * batch_size
        end_idx = min(start_idx + batch_size, len(x_val))
        x_val_batch = x_val[start_idx:end_idx]
        y_val_hard_batch = y_val[start_idx:end_idx]
        y_val_soft_batch = soft_targets_val[start_idx:end_idx]
        
        # Forward pass (no training)
        predictions = kd_student(x_val_batch, training=False)
        
        # Calculate losses
        hard_loss = tf.keras.losses.categorical_crossentropy(y_val_hard_batch, predictions)
        soft_loss = tf.keras.losses.kullback_leibler_divergence(y_val_soft_batch, predictions)
        loss = alpha * hard_loss + (1 - alpha) * soft_loss * (temperature ** 2)
        
        # Update metrics
        val_loss += tf.reduce_mean(loss)
        val_acc += tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(y_val_hard_batch, axis=1), tf.argmax(predictions, axis=1)), 
            tf.float32))
    
    # Average validation metrics
    val_loss /= validation_steps
    val_acc /= validation_steps
    
    # Update history
    kd_history['loss'].append(float(train_loss))
    kd_history['accuracy'].append(float(train_acc))
    kd_history['val_loss'].append(float(val_loss))
    kd_history['val_accuracy'].append(float(val_acc))
    
    # Print epoch results
    print(f"loss: {float(train_loss):.4f} - accuracy: {float(train_acc):.4f} - val_loss: {float(val_loss):.4f} - val_accuracy: {float(val_acc):.4f}")

# Evaluate KD student
kd_loss, kd_acc = kd_student.evaluate(x_test, y_test_one_hot, verbose=0)
print(f"Knowledge distilled student accuracy: {kd_acc*100:.2f}%")

# Print final comparison
print("\nFinal Test Accuracy Comparison:")
print(f"Teacher model:              {teacher_acc*100:.2f}%")
print(f"Student with KD:            {kd_acc*100:.2f}%")
print(f"Student without KD:         {vanilla_acc*100:.2f}%")
print(f"KD Improvement:             {(kd_acc - vanilla_acc)*100:.2f}%")

# Measure inference speed
print("\nMeasuring inference speed and model size...")

# Calculate inference time
def measure_inference_time(model, x_data, batch_size=128, num_runs=5):
    # Warm-up run
    _ = model.predict(x_data[:batch_size], verbose=0)
    
    # Timed runs
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        _ = model.predict(x_data, batch_size=batch_size, verbose=0)
        times.append(time.time() - start_time)
    
    return sum(times) / len(times)  # Average time

# Measure inference times
teacher_time = measure_inference_time(teacher, x_test)
kd_time = measure_inference_time(kd_student, x_test)
vanilla_time = measure_inference_time(vanilla_student, x_test)

print(f"Teacher inference time:     {teacher_time:.4f} seconds")
print(f"Student with KD time:       {kd_time:.4f} seconds")
print(f"Student without KD time:    {vanilla_time:.4f} seconds")
print(f"Speed improvement:          {(teacher_time/kd_time):.2f}x faster than teacher")

# Compare model sizes
teacher_params = teacher.count_params()
kd_params = kd_student.count_params()
vanilla_params = vanilla_student.count_params()

print("\nModel Size Comparison:")
print(f"Teacher parameters:         {teacher_params:,}")
print(f"Student parameters:         {kd_params:,}")
print(f"Size reduction:             {(teacher_params/kd_params):.2f}x smaller than teacher")
print(f"Parameter ratio:            1:{(teacher_params/kd_params):.1f}")

# Visualize results
plt.figure(figsize=(16, 12))

# Training accuracy
plt.subplot(2, 2, 1)
plt.plot(teacher_history.history['accuracy'], label='Teacher (Train)')
plt.plot(teacher_history.history['val_accuracy'], label='Teacher (Val)')
plt.title('Teacher Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Student accuracy comparison
plt.subplot(2, 2, 2)
plt.plot(vanilla_history.history['accuracy'], label='Vanilla Student (Train)')
plt.plot(vanilla_history.history['val_accuracy'], label='Vanilla Student (Val)')
plt.plot(kd_history['accuracy'], label='KD Student (Train)')
plt.plot(kd_history['val_accuracy'], label='KD Student (Val)')
plt.title('Student Models Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Training loss
plt.subplot(2, 2, 3)
plt.plot(teacher_history.history['loss'], label='Teacher (Train)')
plt.plot(teacher_history.history['val_loss'], label='Teacher (Val)')
plt.title('Teacher Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Student loss comparison
plt.subplot(2, 2, 4)
plt.plot(vanilla_history.history['loss'], label='Vanilla Student (Train)')
plt.plot(vanilla_history.history['val_loss'], label='Vanilla Student (Val)')
plt.plot(kd_history['loss'], label='KD Student (Train)')
plt.plot(kd_history['val_loss'], label='KD Student (Val)')
plt.title('Student Models Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('kd_comparison_results.png')
plt.show()

# Create a bar chart comparing accuracies
plt.figure(figsize=(10, 6))
models = ['Teacher', 'Vanilla Student', 'KD Student']
accuracies = [teacher_acc*100, vanilla_acc*100, kd_acc*100]
colors = ['#3498db', '#e74c3c', '#2ecc71']

# Plot bars
bars = plt.bar(models, accuracies, color=colors)
plt.ylabel('Test Accuracy (%)')
plt.title('Model Accuracy Comparison')
plt.ylim(90, 100)  # Focus on the relevant range for MNIST

# Add accuracy values on top of bars
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{acc:.2f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('accuracy_comparison.png')
plt.show()

# Visualize predictions on test examples
def plot_predictions(teacher_model, kd_model, vanilla_model, x_samples, y_true, num_samples=5):
    plt.figure(figsize=(15, 12))
    
    for i in range(num_samples):
        # Get a sample
        img = x_samples[i:i+1]
        true_label = np.argmax(y_true[i])
        
        # Get predictions
        teacher_pred = teacher_model.predict(img, verbose=0)
        kd_pred = kd_model.predict(img, verbose=0)
        vanilla_pred = vanilla_model.predict(img, verbose=0)
        
        # Get predicted classes
        teacher_class = np.argmax(teacher_pred)
        kd_class = np.argmax(kd_pred)
        vanilla_class = np.argmax(vanilla_pred)
        
        # Plot the image
        plt.subplot(num_samples, 4, i*4+1)
        plt.imshow(img[0, :, :, 0], cmap='gray')
        plt.title(f"True: {true_label}")
        plt.axis('off')
        
        # Plot teacher predictions
        plt.subplot(num_samples, 4, i*4+2)
        plt.bar(range(10), teacher_pred[0])
        plt.title(f"Teacher: {teacher_class}")
        plt.ylim(0, 1)
        
        # Plot KD student predictions
        plt.subplot(num_samples, 4, i*4+3)
        plt.bar(range(10), kd_pred[0])
        plt.title(f"KD Student: {kd_class}")
        plt.ylim(0, 1)
        
        # Plot vanilla student predictions
        plt.subplot(num_samples, 4, i*4+4)
        plt.bar(range(10), vanilla_pred[0])
        plt.title(f"Vanilla Student: {vanilla_class}")
        plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('prediction_comparison.png')
    plt.show()

# Get some challenging test samples (not just random ones)
def find_challenging_samples(teacher, vanilla_student, x_test, y_test, n_samples=5):
    # Get predictions
    teacher_preds = teacher.predict(x_test, verbose=0)
    vanilla_preds = vanilla_student.predict(x_test, verbose=0)
    
    # Find samples where teacher is correct but vanilla student is wrong
    teacher_correct = np.argmax(teacher_preds, axis=1) == np.argmax(y_test, axis=1)
    vanilla_correct = np.argmax(vanilla_preds, axis=1) == np.argmax(y_test, axis=1)
    
    # Get indices where teacher is correct but vanilla student is wrong
    interesting_indices = np.where(np.logical_and(teacher_correct, np.logical_not(vanilla_correct)))[0]
    
    # If we don't have enough interesting samples, add some random ones
    if len(interesting_indices) < n_samples:
        random_indices = np.random.choice(
            np.setdiff1d(np.arange(len(x_test)), interesting_indices),
            n_samples - len(interesting_indices),
            replace=False
        )
        selected_indices = np.concatenate([interesting_indices, random_indices])
    else:
        # Take a random subset of the interesting indices
        selected_indices = np.random.choice(interesting_indices, n_samples, replace=False)
    
    return selected_indices

# Find challenging samples
print("Finding challenging test samples for visualization...")
test_indices = find_challenging_samples(teacher, vanilla_student, x_test, y_test_one_hot)
test_samples = x_test[test_indices]
test_labels = y_test_one_hot[test_indices]

# Plot predictions
plot_predictions(teacher, kd_student, vanilla_student, test_samples, test_labels)