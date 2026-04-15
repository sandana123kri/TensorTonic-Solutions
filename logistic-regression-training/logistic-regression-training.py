import numpy as np

def _sigmoid(z):
    return 1 / (1 + np.exp(-z))   # simple sigmoid is fine now

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    X = np.array(X)
    y = np.array(y).astype(float)
    
    N, D = X.shape
    
    w = np.zeros(D)
    b = 0.0
    
    for _ in range(steps):
        # Linear output
        z = X @ w + b
        
        # 🔥 KEY FIX: clip z to avoid overflow
        z = np.clip(z, -20, 20)
        
        # Sigmoid
        p = _sigmoid(z)
        
        # Gradients
        dw = (X.T @ (p - y)) / N
        db = np.mean(p - y)
        
        # Update
        w -= lr * dw
        b -= lr * db
    
    return w, b