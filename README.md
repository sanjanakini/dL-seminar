#  Topic: Backpropagation in ANN, CNN, and RNN

Backpropagation in ANN, CNN, and RNN  
Seminar – Deep Learning

Team Members  
Member 1: NNM23IS161  
Member 2: NNM23IS075  

   **Table of Contents**

1. Introduction to Backpropagation 
2. Backpropagation Overview  
3. Backpropagation in Artificial Neural Networks (ANN)  
4. Backpropagation in Convolutional Neural Networks (CNN)  
5. Backpropagation in Recurrent Neural Networks (RNN)  
6. Differences Between ANN, CNN, and RNN Backpropagation  
7. Code Implementation  
8. Output of Code  
9. References  
10. Conclusion  

## Backpropagation

Backpropagation (Backward Propagation of Errors) is a training algorithm used in neural networks to **minimize the loss function** by updating weights.

It works by:
- Calculating gradients using the **chain rule of calculus**
- Updating weights using **gradient descent**
- Reducing the difference between predicted and actual output

---

## Stages of Backpropagation

### 1. Forward Propagation
- Input passes through the network (input → hidden → output)
- Each neuron applies weights and activation functions
- Produces predicted output

### 2. Loss Calculation
- Compares predicted output with actual output
- Computes error using a loss function (e.g., MSE, Cross-Entropy)

### 3. Backward Propagation
- Error is propagated backward through the network
- Gradients are computed using chain rule
- Weights are updated to minimize loss

---
## ⚙️ Weight Update Rule

Weights are updated using **Gradient Descent**:

---

### Formula

$$
w_{\text{new}} = w_{\text{old}} - \eta \frac{\partial L}{\partial w}
$$

---

### Where:

- **w<sub>new</sub>** → Updated weight  
- **w<sub>old</sub>** → Previous weight  
- **η (eta)** → Learning rate  
- **∂L / ∂w** → Gradient of loss with respect to weight  

---

- The gradient $\frac{\partial L}{\partial w}$ shows how much the loss changes with the weight  
- The learning rate **η** controls how big the update step is  
- Subtracting ensures we move in the direction that **reduces the loss**

---

### In Simple Terms

The model:
1. Finds how wrong it is  
2. Calculates how to fix it  
3. Slightly adjusts weights to improve performance  

Backpropagation helps the model learn from mistakes:
1. Make a prediction  
2. Calculate error  
3. Adjust weights to improve accuracy

Backpropagation relies on the **Chain Rule of Calculus** to compute gradients efficiently.

---

### General Gradient Formula

$$
\frac{\partial L}{\partial w}
= \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial w}
$$

---

### Where:

- **L** → Loss function  
- **w** → Network weight  
- **y** → Output of the neuron  
- **z** → Weighted input (before activation)  

---

The formula breaks down how the loss changes with respect to a weight by splitting it into smaller steps:

- Change of loss w.r.t output → $\frac{\partial L}{\partial y}$  
- Change of output w.r.t weighted input → $\frac{\partial y}{\partial z}$  
- Change of weighted input w.r.t weight → $\frac{\partial z}{\partial w}$  

These are multiplied together using the **chain rule** to compute the final gradient.

---

### In Simple Terms

Backpropagation:
1. Tracks how error flows backward  
2. Computes gradients step-by-step  
3. Updates weights to reduce loss

     ##  1) Backpropagation in Artificial Neural Networks (ANN)

### Architecture

Artificial Neural Networks (ANNs) consist of **fully connected layers**, where each neuron in one layer is linked to every neuron in the next layer through weights and biases.

---

### Forward Propagation

Each neuron performs a linear operation:

$$
z = Wx + b
$$

Where:
- **W** → Weight  
- **x** → Input  
- **b** → Bias  

The result is passed through an activation function such as **sigmoid**:

$$
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

---

### Loss Function

To evaluate prediction accuracy, we use **Mean Squared Error (MSE)**:

$$
L = \frac{1}{2}(y - \hat{y})^2
$$

Where:
- **y** → Actual output  
- **ŷ** → Predicted output  

---

### Gradient Computation

Using the **chain rule**, gradients are calculated as:

$$
\frac{\partial L}{\partial W}
= \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial W}
$$

This helps determine how changes in weights affect the loss.

---

###  Algorithm (ANN Backpropagation)

1. Initialize weights and biases randomly  
2. Perform forward propagation  
3. Compute the loss  
4. Calculate gradients using backpropagation  
5. Update weights using gradient descent  
6. Repeat until the model improves  

---
###  Example Calculation (New Values)

**Given:**
- Input: $x = 2$  
- Weight: $W = 0.3$  
- Bias: $b = 0.1$  

---

**Step 1: Linear computation**

$$
z = (0.3)(2) + 0.1 = 0.7
$$

---

**Step 2: Apply sigmoid activation**

$$
\hat{y} = \frac{1}{1 + e^{-0.7}} \approx 0.67
$$

---

**Step 3: Compute loss**  
(Assume actual output $y = 1$)

$$
L = \frac{1}{2}(1 - 0.67)^2 \approx 0.054
$$


The computed loss is then used in backpropagation to adjust the weight **W**. By repeatedly updating weights, the network gradually reduces error and improves its predictions.

##  2) Backpropagation in Convolutional Neural Networks (CNN)

###  Architecture

Convolutional Neural Networks (CNNs) are specifically designed for **spatial data** like images.

Unlike fully connected networks, CNNs use:
- Convolutional layers  
- Filters (kernels)  
- Feature maps  
- Pooling layers  

These components help the network capture spatial patterns such as edges, textures, and shapes.

---

###  Convolution Operation

Feature maps are created by sliding a kernel over the input image:

$$
F(i, j) = \sum_m \sum_n I(i+m, j+n)\, K(m,n)
$$

Where:
- **I** → Input image  
- **K** → Kernel (filter)  
- **F** → Output feature map  

---

###  Gradient Computation

During backpropagation, gradients are calculated for each kernel:

$$
\frac{\partial L}{\partial K} = I * \delta
$$

Where:
- **I** → Input image  
- **δ (delta)** → Error gradient from the next layer  

---

###  Kernel Update Rule

$$
K_{\text{new}} = K_{\text{old}} - \eta \frac{\partial L}{\partial K}
$$

Where:
- **η (eta)** → Learning rate  

---

###  Algorithm (CNN Backpropagation)

1. Apply convolution filters to the input image  
2. Generate feature maps  
3. Apply activation functions  
4. Compute the loss  
5. Calculate gradients for each kernel  
6. Update kernel values using gradient descent  
7. Repeat until the model improves  

---

###  Example Calculation

**Input Image:**

$$
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{bmatrix}
$$

**Kernel:**

$$
\begin{bmatrix}
1 & 0 \\
0 & -1
\end{bmatrix}
$$

---

**Step 1: Convolution (top-left region)**

$$
(1 \times 1) + (2 \times 0) + (4 \times 0) + (5 \times -1) = -4
$$

---

**Step 2: Feature Map**

$$
\begin{bmatrix}
-4 & -4 \\
-4 & -4
\end{bmatrix}
$$

---

###  Final Insight

Backpropagation adjusts the kernel values based on the error gradient. Over multiple iterations, the filters learn to detect important features in the input image, improving model performance.
##  3) Backpropagation in Recurrent Neural Networks (RNN)

###  Architecture

Recurrent Neural Networks (RNNs) are designed to handle **sequential data**, such as:
- Text  
- Speech  
- Time series  

They use a **hidden state** to retain information from previous time steps, allowing the model to learn temporal patterns.

---

###  Hidden State Update

At each time step $t$, the hidden state is updated as:

$$
h_t = \tanh(Wx_t + U h_{t-1} + b)
$$

Where:
- **$x_t$** → Input at time step $t$  
- **$h_{t-1}$** → Previous hidden state  
- **$W$** → Input weight matrix  
- **$U$** → Recurrent weight matrix  
- **$b$** → Bias  

---

###  Output Equation

$$
y_t = V h_t
$$

Where:
- **$V$** → Output weight matrix  

---

###  Loss Function

The total loss over the entire sequence is:

$$
L = \sum_{t=1}^{T} (y_t - \hat{y}_t)^2
$$

---

###  Backpropagation Through Time (BPTT)

RNNs are trained using **Backpropagation Through Time (BPTT)**, where the network is unrolled across time steps and gradients are propagated backward:

$$
\frac{\partial L}{\partial W} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W}
$$

This means the contribution of each time step is accumulated when updating weights.

---

###  Algorithm (RNN Backpropagation)

1. Input the sequence step-by-step  
2. Compute hidden states at each time step  
3. Generate outputs  
4. Calculate loss for each step  
5. Unroll the network across time  
6. Backpropagate errors through time (BPTT)  
7. Update weights using gradient descent  

---

###  Example (Sequence Processing)

**Input sequence:**
- $x_1 = 1$  
- $x_2 = 2$  
- $x_3 = 3$  

**Hidden state update:**

$$
h_t = \tanh(Wx_t + U h_{t-1})
$$

- At each step, the hidden state depends on both the current input and the previous state  
- Information flows across time steps  

---

###  Final Insight

In RNNs, each time step contributes to the total loss. During backpropagation, gradients from all time steps are combined, allowing the model to learn patterns over sequences.

##  Backpropagation Comparison: ANN vs CNN vs RNN

| Feature | ANN | CNN | RNN |
|--------|-----|-----|-----|
| **Architecture Style** | Dense (fully connected layers) | Uses convolutional layers with kernels | Uses recurrent connections with memory |
| **Typical Data** | Structured/tabular data | Visual data (images) | Sequential data (text, time-series) |
| **Gradient Flow** | Propagates across layers | Flows through filters and feature maps | Propagates across time steps |
| **Parameter Sharing** | No sharing of weights | Same filters reused across input | Same weights reused across time |
| **Training Approach** | Standard backpropagation | Backpropagation adapted for convolution layers | Backpropagation Through Time (BPTT) |

---

##  Outputs of Code

###  ANN Output


Prediction: [[0.66818777]]  
Loss: [[0.05504968]]  
Updated Weight: [[0.31471341]]  
Updated Bias: [[0.1073567]]

---

- The model predicts a value of approximately **0.66** using the **sigmoid activation function**.  
- Since the actual (target) output is **1**, there is a small difference between predicted and actual values.  
- This difference is quantified as the **loss**, which is approximately **0.055**.  
- During backpropagation, gradients are calculated to understand how the error changes with respect to weights.  
- The weight is updated from **0.3 → 0.314** to reduce the error.  
- The bias is also updated accordingly to further improve the prediction.

## In simple words,
- Prediction ≠ actual → error exists
- Backpropagation reduces this error
- Weight increased → model trying to improve prediction
  
###  CNN Output

```
Feature Map:
[[-3, -3],
 [-3, -3]]

Updated Kernel:
[[0.96, 0.05],
 [0.05, -0.95]]
```
- The convolution operation extracts features from the image using a kernel
- The feature map values are all -4, showing how the filter responds to the image.
- The loss is computed as the sum of feature values.
- Backpropagation updates the kernel values to improve feature extraction.

  ## In simple words,
- Kernel detects patterns
- Negative values → specific feature response
- Kernel updated → better feature detection next time
  ### 🔁 RNN Output

```
Hidden States:
[0.46, 0.79, 0.93]

Loss:
2.19

Updated W:
0.48

Updated U:
0.20
```
- The RNN processes input sequentially and stores information in hidden states.
- Each hidden state depends on both current input and previous state.
- The loss is computed across all time steps.
- Using Backpropagation Through Time, weights W and U are updated.

  ## In simple words,
- Hidden state remembers past information
- Gradients flow back through time
- Both weights updated → sequence learning improves

- SO,In all three models, backpropagation computes gradients and updates parameters to reduce loss and improve predictions.

  ## 📚 References

- GeeksforGeeks. *Backpropagation in Neural Networks*.  
  Available at: https://www.geeksforgeeks.org/backpropagation-in-neural-network/

- GeeksforGeeks. *Convolutional Neural Networks (CNN)*.  
  Available at: https://www.geeksforgeeks.org/convolutional-neural-network-cnn/

- GeeksforGeeks. *Recurrent Neural Networks (RNN)*.  
  Available at: https://www.geeksforgeeks.org/recurrent-neural-network-rnn/

  ##  Conclusion

Backpropagation serves as the fundamental learning mechanism in neural networks, enabling models to adjust their parameters by minimizing error.

While the core idea remains consistent, its application differs across architectures:

- **ANN** applies standard backpropagation across fully connected layers  
- **CNN** modifies the process to update convolutional filters for spatial feature extraction  
- **RNN** extends backpropagation across time steps using Backpropagation Through Time (BPTT)  

Understanding these variations is important for selecting the most suitable model based on the type of data and problem being addressed.
