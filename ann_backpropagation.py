import numpy as np

# Sigmoid activation function
# Converts input into a value between 0 and 1 (used for prediction)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
# Required during backpropagation to compute gradients
def sigmoid_derivative(output):
    return output * (1 - output)


# Step 1: Input and parameters

x = np.array([[2]])      # Input value
y = np.array([[1]])      # Actual/target output

W = np.array([[0.3]])    # Initial weight
b = np.array([[0.1]])    # Bias term

learning_rate = 0.1      # Controls how much weights change


# Step 2: Forward Propagation
# Linear combination of input and weight
z = np.dot(x, W) + b     # z = Wx + b

# Apply activation function to get prediction
y_hat = sigmoid(z)

# Compute error using Mean Squared Error
loss = 0.5 * (y - y_hat) ** 2

print("Prediction:", y_hat)
print("Loss:", loss)


# Step 3: Backpropagation

# Step 3.1: How loss changes w.r.t prediction
dL_dyhat = -(y - y_hat)

# Step 3.2: How prediction changes w.r.t z
dyhat_dz = sigmoid_derivative(y_hat)

# Step 3.3: How z changes w.r.t weight
dz_dW = x

# Chain rule: combine all derivatives
# This tells how loss changes w.r.t weight
dL_dW = dL_dyhat * dyhat_dz * dz_dW

# Gradient w.r.t bias (since dz/db = 1)
dL_db = dL_dyhat * dyhat_dz


# Step 4: Update weights
# Move weights in direction that reduces loss
W = W - learning_rate * dL_dW
b = b - learning_rate * dL_db

print("Updated Weight:", W)
print("Updated Bias:", b)