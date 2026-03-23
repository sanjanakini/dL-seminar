import numpy as np

# Input image (3x3 matrix)
image = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Kernel (filter used to extract features)
kernel = np.array([
    [1, 0],
    [0, -1]
])

learning_rate = 0.1


# Step 1: Convolution operation
def convolve(img, ker):
    output = np.zeros((2, 2))  # Output feature map size
    
    for i in range(2):
        for j in range(2):
            # Extract 2x2 region from image
            region = img[i:i+2, j:j+2]
            
            # Multiply region with kernel and sum
            # This detects patterns (edges, textures)
            output[i, j] = np.sum(region * ker)
    
    return output

# Generate feature map
feature_map = convolve(image, kernel)
print("Feature Map:\n", feature_map)

# Step 2: Loss calculation

# Simplified loss (sum of all values)
loss = np.sum(feature_map)
print("Loss:", loss)


# Step 3: Backpropagation
# Initialize gradient for kernel
dL_dK = np.zeros_like(kernel)

for i in range(2):
    for j in range(2):
        # Same regions used during forward pass
        region = image[i:i+2, j:j+2]
        
        # Accumulate gradients
        # Shows how each kernel value affects loss
        dL_dK += region

# Step 4: Update kernel
kernel = kernel - learning_rate * dL_dK

print("Updated Kernel:\n", kernel)