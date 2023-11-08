from abc import ABC, abstractmethod
import math
import numpy as np

class Regressor(ABC):
    @abstractmethod
    def fit(self, x_sample: np.ndarray, y_sample: np.ndarray):
        """サンプルに合わせて内部のパラメータを学習する

        Args:
            x_sample (np.ndarray): サンプルのx（1次元配列）
            y_sample (np.ndarray): サンプルのy（1次元配列）
        """
        ...

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """与えられたxに対する関数値を予測する

        Args:
            x (np.ndarray): 計算すべきxの値（1次元配列）

        Returns:
            np.ndarray: 予測値。入力xに対応した1次元配列
        """
        ...

class PolyRegressor(Regressor):
    def __init__(self, d):
        self.d = d
        self.p = np.arange(d+1)[np.newaxis, :] # 1 x (d+1)

    def fit(self, x_sample, y_sample):
        X_sample = x_sample[:, np.newaxis] ** self.p
        XX_inv_sample = np.linalg.inv(X_sample.T @ X_sample)
        self.a =  XX_inv_sample @ X_sample.T @ y_sample[:, np.newaxis]

    def predict(self, x):
        X = x[:, np.newaxis] ** self.p
        y_pred = np.squeeze(X @ self.a)
        return y_pred

class GPRegressor(Regressor):
    def __init__(self, sigma_x, sigma_y):
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def fit(self, x_sample: np.ndarray, y_sample: np.ndarray):
        x_s = x_sample[:, np.newaxis]
        y_s = y_sample[:, np.newaxis]
        G = self._gaussian(x_s, x_s.T)
        sigma_I = self.sigma_y * np.eye(G.shape[0])
        self.x_s = x_s
        self.a = np.linalg.inv(G + sigma_I) @ y_s

    def predict(self, x):
        g = self._gaussian(x[:, np.newaxis], self.x_s.T)
        y_pred = np.squeeze(g @ self.a)
        return y_pred

    def _gaussian(self, col, row) -> np.ndarray:
        return np.exp(- (col - row) ** 2 / (2 * self.sigma_x ** 2))

class Layer(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """順伝播の計算をする

        Args:
            x (np.ndarray): 層の入力

        Returns:
            np.ndarray: 計算結果
        """
        ...

    @abstractmethod
    def backward(self, dL_dy: np.ndarray, learning_rate: float) -> np.ndarray:
        """勾配を逆伝播しつつパラメータ更新する

        Args:
            dL_dy (np.ndarray): 逆伝播されてきた勾配
            learning_rate (float): パラメータ更新のための学習率

        Returns:
            np.ndarray: この層の入力に対する損失関数の勾配
        """
        ...

class FCLayer(Layer):
    def __init__(self, d_in, d_out):
        gain = math.sqrt(2) # ReLUを使うため
        scale = gain / math.sqrt(d_in) # Heの初期化（fan_in）
        self.weight = np.random.normal(0.0, scale, size=(d_out, d_in))
        self.bias = np.zeros((d_out, 1))

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x.copy()
        return self.weight @ x + self.bias

    def backward(self, dL_dy: np.ndarray, learning_rate: float) -> np.ndarray:
        dL_dx = self.weight.T @ dL_dy
        dL_db = dL_dy.copy()
        self.bias = self.bias - learning_rate * dL_db
        dL_dW = dL_dy @ self.x.T
        self.weight = self.weight - learning_rate * dL_dW
        return dL_dx

class ReLULayer(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x.copy()
        return np.where(x > 0, x, 0.0)

    def backward(self, dL_dy: np.ndarray, learning_rate: float) -> np.ndarray:
        return np.where(self.x > 0, dL_dy, 0.0)

class NNRegressor(Regressor):
    def __init__(self, n_layer, n_dim, learning_rate, epoch):
        self.n_layer = n_layer
        self.n_dim = n_dim
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.layers: list[Layer] = []
        for i in range(self.n_layer):
            if i == 0:
                d_in = 1
            else:
                d_in = self.n_dim
            if i == self.n_layer-1:
                d_out = 1
            else:
                d_out = self.n_dim
            self.layers.append(FCLayer(d_in, d_out))
            if i < self.n_layer-1:
                self.layers.append(ReLULayer())

    def fit(self, x_sample: np.ndarray, y_sample: np.ndarray):
        for _ in range(self.epoch):
            for x, y in zip(x_sample, y_sample):
                x = np.ones((1, 1)) * x
                y_pred = self._forward(x)
                dL_dy = y_pred - y
                self._backward(dL_dy)

    def _forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def _backward(self, dL_dy):
        for layer in self.layers[::-1]:
            dL_dy = layer.backward(dL_dy, self.learning_rate)

    def predict(self, x):
        z = x[np.newaxis, :]
        for layer in self.layers:
            z = layer.forward(z)
        y_pred = np.squeeze(z)
        return y_pred

def build_regressor(name, kwargs_all) -> Regressor:
    REGRESSORS = dict(
        poly=PolyRegressor,
        gp=GPRegressor,
        nn=NNRegressor,
    )
    regressor_cls = REGRESSORS[name]
    kwargs = kwargs_all[name]
    return regressor_cls(**kwargs)