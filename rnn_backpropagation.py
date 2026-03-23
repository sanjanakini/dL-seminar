import numpy as np

# Activation function
# tanh is used to keep values between -1 and 1
def tanh(x):
    return np.tanh(x)

# Step 1: Input sequence
x = [1, 2, 3]   # Sequential input

# Initialize weights
W = 0.5   # Input weight
U = 0.2   # Recurrent weight (previous state influence)

learning_rate = 0.01

h_prev = 0        # Initial hidden state
hidden_states = []  # Store all hidden states

# Step 2: Forward pass (through time)
for t in range(len(x)):
    # Compute new hidden state
    # Depends on current input and previous state
    h = tanh(W * x[t] + U * h_prev)
    
    hidden_states.append(h)
    h_prev = h  # Update for next time step

print("Hidden States:", hidden_states)

# Step 3: Loss calculation

# Simplified loss: sum of hidden states
loss = sum(hidden_states)
print("Loss:", loss)

# Step 4: Backpropagation Through Time (BPTT)
dW = 0
dU = 0
dh_next = 0

# Go backward through time steps
for t in reversed(range(len(x))):
    
    # Derivative of tanh
    dh = (1 - hidden_states[t]**2) + dh_next
    
    # Accumulate gradients
    dW += dh * x[t]
    dU += dh * (hidden_states[t-1] if t > 0 else 0)
    
    # Pass gradient backward in time
    dh_next = dh * U


# Step 5: Update weights
W = W - learning_rate * dW
U = U - learning_rate * dU

print("Updated W:", W)
print("Updated U:", U)