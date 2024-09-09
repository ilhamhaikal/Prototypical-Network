import matplotlib.pyplot as plt

# Data
epochs = list(range(1, 21))
train_loss = [
    1.5056, 1.2508, 1.0863, 0.9698, 0.9207, 0.8272,
    0.7712, 0.7033, 0.6500, 0.5987, 0.4797, 0.3979,
    0.3808, 0.3227, 0.2972, 0.2993, 0.2190, 0.2435,
    0.2297, 0.2176
]
train_accuracy = [
    0.7333, 0.7531, 0.7632, 0.7867, 0.8250, 0.8267,
    0.8267, 0.8667, 0.8667, 0.8228, 0.8205, 0.8158,
    0.8833, 0.9000, 0.8833, 0.9103, 0.9000, 0.9167,
    0.9167, 0.9221
]
val_loss = [
    1.4267, 1.0582, 0.8703, 0.7253, 0.6182, 0.5438,
    0.4951, 0.4586, 0.4308, 0.3668, 0.3347, 0.2987,
    0.2862, 0.2757, 0.2657, 0.2576, 0.2488, 0.2406,
    0.2336, 0.2285
]
val_accuracy = [
    0.5000, 0.5677, 0.5899, 0.6000, 0.6165, 0.6667,
    0.7099, 0.7299, 0.7569, 0.7677, 0.7990, 0.8333,
    0.8444, 0.8699, 0.8833, 0.9000, 0.9166, 0.9269,
    0.9363, 0.9444
]

# Plotting
plt.figure(figsize=(12, 6))

# Plot train loss and validation loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, marker='o', label='Train Loss')
plt.plot(epochs, val_loss, marker='o', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss vs Validation Loss')
plt.legend()
plt.grid(True)

# Plot train accuracy and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracy, marker='o', label='Train Accuracy')
plt.plot(epochs, val_accuracy, marker='o', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train Accuracy vs Validation Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()