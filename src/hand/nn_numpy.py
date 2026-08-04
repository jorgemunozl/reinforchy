import numpy as np


class NeuralNetwork:
    """Fully-connected network with ReLU hidden layers and a linear output."""

    def __init__(self, sizes, lr=0.01, seed=42):
        rng = np.random.default_rng(seed)
        self.sizes = sizes
        self.lr = lr

        # He initialization (good for ReLU).
        self.W = []
        self.b = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            self.W.append(
                rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)
            )
            self.b.append(np.zeros(fan_out))

    def forward(self, X):
        """Propagate X through the network, caching activations for backprop."""
        self.z = []  # pre-activations of each layer
        self.a = [X]  # activations; a[i] is the input to layer i
        for W, b in zip(self.W, self.b):
            z = self.a[-1] @ W + b
            self.z.append(z)
            self.a.append(np.maximum(0.0, z))  # ReLU
        self.a[-1] = self.z[-1]  # linear output layer
        return self.a[-1]

    def backward(self, X, y):
        """Compute gradients of the mean-squared-error loss (no update yet)."""
        n = X.shape[0]
        pred = self.a[-1]

        # dL/dz for the linear output layer, L = mean((pred - y)^2).
        dz = 2.0 * (pred - y) / n

        dW = [None] * len(self.W)
        db = [None] * len(self.b)
        for i in range(len(self.W) - 1, -1, -1):
            dW[i] = self.a[i].T @ dz
            db[i] = dz.sum(axis=0)
            if i > 0:
                # Chain rule through ReLU: derivative is 1 where z > 0.
                dz = (dz @ self.W[i].T) * (self.z[i - 1] > 0)
        return dW, db

    def train(self, X, y, epochs=2000, batch_size=32):
        n = X.shape[0]
        interval = max(1, epochs // 10)
        for epoch in range(epochs):
            idx = np.random.permutation(n)  # shuffle once per epoch
            for start in range(0, n, batch_size):
                batch = idx[start : start + batch_size]
                Xb, yb = X[batch], y[batch]
                self.forward(Xb)
                dW, db = self.backward(Xb, yb)
                for i in range(len(self.W)):
                    self.W[i] -= self.lr * dW[i]
                    self.b[i] -= self.lr * db[i]
            if epoch % interval == 0 or epoch == epochs - 1:
                print(f"epoch {epoch:5d}  loss {self.loss(X, y):.6f}")

    def loss(self, X, y):
        pred = self.forward(X)
        return np.mean((pred - y) ** 2)

    def predict(self, X):
        return self.forward(X)


def main():
    rng = np.random.default_rng(0)

    # Training data: y = sin(x) plus a little noise.
    X = np.linspace(-2 * np.pi, 2 * np.pi, 500).reshape(-1, 1)
    y = np.sin(X) + 0.05 * rng.standard_normal(X.shape)

    net = NeuralNetwork(sizes=[1, 16, 16, 1], lr=0.01, seed=42)
    net.train(X, y, epochs=2000, batch_size=32)

    # Evaluate on a clean grid (no noise).
    X_test = np.linspace(-2 * np.pi, 2 * np.pi, 200).reshape(-1, 1)
    y_true = np.sin(X_test)
    y_pred = net.predict(X_test)
    mae = np.mean(np.abs(y_pred - y_true))
    print(f"\ntest MAE vs sin(x): {mae:.5f}")
    print("\n     x      sin(x)  predicted")
    for x, t, p in zip(X_test[::25, 0], y_true[::25, 0], y_pred[::25, 0]):
        print(f"{x:8.4f}  {t:8.4f}  {p:8.4f}")


if __name__ == "__main__":
    main()
