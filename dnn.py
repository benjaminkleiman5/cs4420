import numpy as np
import pickle

def compute_softmax(x):
    """Compute softmax transformation in a numerically stable way.
    Args:
    x: logits, of the shape [d, batch]
    Returns:
    Softmax probability of the input logits, of the shape [d, batch].
    """
    out = x - np.max(x, 0, keepdims=True)
    out_exp = np.exp(out)
    exp_sum = np.sum(out_exp, 0, keepdims=True)
    probs = out_exp / exp_sum
    return probs

def compute_ce_loss(out, y, loss_mask):
    """Compute cross-entropy loss, averaged over valid training samples.
    Args:
    out: dnn output, of shape [dout, batch].
    y: integer labels, [batch].
    loss_mask: loss mask of shape [batch], 1.0 for valid sample, 0.0 for padded
    sample.
    Returns:
    Cross entropy loss averaged over valid samples, and gradient wrt output.
    """
    pred_probs = compute_softmax(out)
    ce_loss = 0.0
    grad_out = np.zeros_like(out)
    for k in range(len(y)):
        if loss_mask[k] != 1.0:
            continue
        correct_class = y[k]
        ce_loss -= np.log(pred_probs[correct_class, k])
        grad_out[:, k] = pred_probs[:, k]
        grad_out[correct_class, k] -= 1
    valid_sample_count = np.sum(loss_mask)
    if valid_sample_count > 0:
        ce_loss /= valid_sample_count  # Average the loss only
    return ce_loss, grad_out

class FeedForwardNetwork:
    
    def __init__(self, din, dout, num_hidden_layers, hidden_layer_width):
        self.din = din
        self.dout = dout
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_width = hidden_layer_width
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        # Input layer to first hidden layer
        self.weights.append(np.random.uniform(-0.05, 0.05, (self.hidden_layer_width, self.din)))
        self.biases.append(np.zeros((self.hidden_layer_width)))
        # Hidden layers
        for index in range(self.num_hidden_layers - 1):
            self.weights.append(np.random.uniform(-0.05, 0.05, (self.hidden_layer_width, self.hidden_layer_width)))
            self.biases.append(np.zeros((self.hidden_layer_width)))
        # Last hidden layer to output layer
        self.weights.append(np.random.uniform(-0.05, 0.05, (self.dout, self.hidden_layer_width)))
        self.biases.append(np.zeros((self.dout)))
        
    def forward(self, x):
        """Forward the feedforward neural network.
        Args:
        x: shape [din, batch].
        Returns:
        Output of shape [dout, batch], and a list of hidden layer activations,
        each of the shape [hidden_layer_width, batch].
        """
        hidden_activations = []
        input_data = x  # Input data to the first hidden layer
        # Pass through all hidden layers
        for i in range(self.num_hidden_layers):
            # Linear transformation
            z = np.dot(self.weights[i], input_data) + self.biases[i][:, np.newaxis]
            # Apply ReLU activation
            activation = np.maximum(0, z)
            # Store activations
            hidden_activations.append(activation)
            # Set input for the next layer
            input_data = activation
        # Output layer (no activation applied)
        output = (np.dot(self.weights[-1], input_data) + self.biases[-1][:, np.newaxis])
        return output, hidden_activations

    def activation_deriv(self, x):
        return np.where(x > 0, 1, 0)
    
    def backward(self, x, hidden, loss_grad, loss_mask):
        """Backpropagation of feedforward neural network.
        Args:
        x: input, of shape [din, batch].
        hidden: list of hidden activations, each of the shape
        [hidden_layer_width, batch].
        loss_grad: gradient with respect to out, of shape [dout, batch].
        loss_mask: loss mask of shape [batch], 1.0 for valid sample, 0.0 for
        padded sample.
        Returns:
        Returns gradient, averaged over valid samples, with respect to weights
        and biases.
        """
        valid_sample_count = np.sum(loss_mask)

        weight_grads = [np.zeros_like(w) for w in self.weights]
        bias_grads = [np.zeros_like(b) for b in self.biases]

        # Output layer gradients: loss_grad represents (predicted_probs - true_labels)
        delta = loss_grad * loss_mask[np.newaxis, :]
        weight_grads[-1] = np.dot(delta, hidden[-1].T) / valid_sample_count
        bias_grads[-1] = np.sum(delta, axis=1) / valid_sample_count

        # Backprop through hidden layers
        for i in range(self.num_hidden_layers - 1, -1, -1):
            relu_grad = self.activation_deriv(hidden[i])
            delta = np.dot(self.weights[i + 1].T, delta) * relu_grad
            delta *= loss_mask[np.newaxis, :]
            
            if i == 0:
                weight_grads[i] = np.dot(delta, x.T) / valid_sample_count
            else:
                weight_grads[i] = np.dot(delta, hidden[i - 1].T) / valid_sample_count

            bias_grads[i] = np.sum(delta, axis=1) / valid_sample_count

        return weight_grads, bias_grads
        
    def update_model(self, w_updates, b_updates):
        """Update the weights and biases of the model.
        Args:
        w_updates: a list of updates to each weight matrix.
        b_updates: a list of updates to weight bias vector.
        """
        self.weights = [w + u for w, u in zip(self.weights, w_updates)]
        self.biases = [b + u for b, u in zip(self.biases, b_updates)]
        
    def predict(self, x):
        """Compute predictions on a minibath.
        Args:
        x: input, of shape [din, batch].
        Returns:
        The discrete model predictions and the probabilities of predicting each
        class.
        """
        out, _ = self.forward(x)
        probs = compute_softmax(out)
        return np.argmax(out, 0), probs

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({'weights': self.weights, 'biases': self.biases}, f)
            
    def restore_model(self, filename):
        with open(filename, 'rb') as f:
            loaded_dict = pickle.load(f)
            self.weights = loaded_dict['weights']
            self.biases = loaded_dict['biases']






