import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Extract the weights from the provided dictionary
weights = {
    'base_model.model.bert.encoder.layer.0.attention.self.query.lora_B.weight': [62, 16, 35, 26, 58, 6, 14, 60],
    'base_model.model.bert.encoder.layer.0.attention.self.key.lora_B.weight': [22, 61, 5, 35, 48, 49, 47, 53],
    'base_model.model.bert.encoder.layer.0.attention.self.value.lora_B.weight': [26, 17, 15, 43, 46, 28, 25, 63],
    'base_model.model.bert.encoder.layer.1.attention.self.query.lora_B.weight': [1, 7, 42, 28, 27, 40, 58, 51],
    'base_model.model.bert.encoder.layer.1.attention.self.key.lora_B.weight': [7, 2, 6, 16, 31, 44, 57, 20],
    'base_model.model.bert.encoder.layer.1.attention.self.value.lora_B.weight': [49, 1, 22, 14, 0, 34, 62, 42],
    'base_model.model.bert.encoder.layer.2.attention.self.query.lora_B.weight': [36, 39, 20, 11, 18, 54, 49, 13],
    'base_model.model.bert.encoder.layer.2.attention.self.key.lora_B.weight': [13, 20, 48, 33, 45, 14, 11, 62],
    'base_model.model.bert.encoder.layer.2.attention.self.value.lora_B.weight': [53, 59, 48, 3, 4, 27, 32, 45],
    'base_model.model.bert.encoder.layer.3.attention.self.query.lora_B.weight': [51, 11, 60, 34, 17, 29, 19, 42],
    'base_model.model.bert.encoder.layer.3.attention.self.key.lora_B.weight': [61, 11, 60, 30, 29, 33, 23, 40],
    'base_model.model.bert.encoder.layer.3.attention.self.value.lora_B.weight': [2, 4, 57, 36, 58, 7, 9, 45],
    'base_model.model.bert.encoder.layer.4.attention.self.query.lora_B.weight': [53, 3, 46, 8, 50, 34, 4, 25],
    'base_model.model.bert.encoder.layer.4.attention.self.key.lora_B.weight': [3, 50, 53, 6, 9, 24, 10, 16],
    'base_model.model.bert.encoder.layer.4.attention.self.value.lora_B.weight': [60, 20, 18, 63, 55, 38, 40, 49],
    'base_model.model.bert.encoder.layer.5.attention.self.query.lora_B.weight': [18, 36, 56, 9, 15, 42, 26, 20],
    'base_model.model.bert.encoder.layer.5.attention.self.key.lora_B.weight': [3, 40, 26, 55, 56, 61, 54, 51],
    'base_model.model.bert.encoder.layer.5.attention.self.value.lora_B.weight': [32, 40, 24, 55, 28, 18, 1, 53],
    'base_model.model.bert.encoder.layer.6.attention.self.query.lora_B.weight': [47, 13, 59, 12, 27, 8, 28, 2],
    'base_model.model.bert.encoder.layer.6.attention.self.key.lora_B.weight': [47, 12, 55, 27, 30, 63, 5, 62],
    'base_model.model.bert.encoder.layer.6.attention.self.value.lora_B.weight': [48, 3, 57, 45, 5, 21, 24, 32],
    'base_model.model.bert.encoder.layer.7.attention.self.query.lora_B.weight': [4, 62, 48, 23, 9, 35, 46, 33],
    'base_model.model.bert.encoder.layer.7.attention.self.key.lora_B.weight': [1, 62, 11, 48, 63, 21, 18, 56],
    'base_model.model.bert.encoder.layer.7.attention.self.value.lora_B.weight': [22, 7, 34, 5, 10, 47, 16, 63],
    'base_model.model.bert.encoder.layer.8.attention.self.query.lora_B.weight': [28, 29, 54, 5, 52, 50, 18, 12],
    'base_model.model.bert.encoder.layer.8.attention.self.key.lora_B.weight': [16, 28, 15, 61, 21, 53, 7, 2],
    'base_model.model.bert.encoder.layer.8.attention.self.value.lora_B.weight': [11, 12, 51, 46, 48, 14, 31, 35],
    'base_model.model.bert.encoder.layer.9.attention.self.query.lora_B.weight': [36, 14, 56, 37, 29, 7, 6, 23],
    'base_model.model.bert.encoder.layer.9.attention.self.key.lora_B.weight': [18, 63, 14, 43, 10, 36, 37, 42],
    'base_model.model.bert.encoder.layer.9.attention.self.value.lora_B.weight': [55, 24, 15, 48, 30, 33, 34, 19],
    'base_model.model.bert.encoder.layer.10.attention.self.query.lora_B.weight': [10, 2, 0, 23, 17, 43, 63, 9],
    'base_model.model.bert.encoder.layer.10.attention.self.key.lora_B.weight': [10, 46, 43, 36, 17, 6, 18, 60],
    'base_model.model.bert.encoder.layer.10.attention.self.value.lora_B.weight': [4, 42, 30, 16, 18, 6, 34, 41],
    'base_model.model.bert.encoder.layer.11.attention.self.query.lora_B.weight': [9, 28, 1, 50, 7, 44, 14, 4],
    'base_model.model.bert.encoder.layer.11.attention.self.key.lora_B.weight': [57, 14, 9, 28, 3, 17, 22, 7],
    'base_model.model.bert.encoder.layer.11.attention.self.value.lora_B.weight': [52, 16, 17, 47, 24, 39, 44, 38]
}

# Prepare the grid data for visualization
num_layers = len(weights)
binary_grid = np.zeros((64, num_layers))

# Fill the binary grid
for i, (layer, values) in enumerate(weights.items()):
    for value in values:
        binary_grid[value, i] = 1

# Shorten the layer names by omitting ".lora_B.weight"
short_layers = [layer.split('base_model.model.bert.encoder.')[-1] for layer in weights.keys()]
short_layers_final = [layer.replace('.attention.self.query.lora_B.weight', '.query')
                      .replace('.attention.self.key.lora_B.weight', '.key')
                      .replace('.attention.self.value.lora_B.weight', '.value') for layer in short_layers]

# Create a rotated heatmap with the binary grid, further shortened layer names, no colorbar, and square grids
plt.figure(figsize=(20, 20))
sns.heatmap(binary_grid.T, annot=False, cmap="Blues", linewidths=.5, cbar=False, square=True)

# Add labels and title
plt.ylabel('Layers')
plt.xlabel('Selected Ranks')
plt.title('Binary Heatmap of lora_B Weights across BERT Encoder Layers (Rotated)')

# Adding further shortened layer names to y-axis
plt.yticks(ticks=np.arange(num_layers) + 0.5, labels=short_layers_final, rotation=0)
plt.tight_layout()
plt.show()
