import numpy as np
import matplotlib.pyplot as plt


class LDA:
    """
    Linear Discriminant Analysis (LDA) classifier.
    """
    X0: np.ndarray  # data points of positive class
    X1: np.ndarray  # data points of negative class
    w: np.ndarray   # projection line, w = (w1, w2), normalized
    
    def __init__(self):
        self.X0 = None
        self.X1 = None
        self.w = None

    def fit(self, X_train, y_train):
        self.X0 = X_train[y_train == 0]
        self.X1 = X_train[y_train == 1]
        Sw = np.cov(self.X0.T) + np.cov(self.X1.T)
        self.w = np.linalg.inv(Sw).dot(self.mu0 - self.mu1)
        self.w /= np.linalg.norm(self.w)  # normalize w
        self.__plot()

    def predict(self, X):
        proj = X.dot(self.w)
        proj0 = self.mu0.dot(self.w)
        proj1 = self.mu1.dot(self.w)
        return np.where(np.abs(proj - proj0) < np.abs(proj - proj1), 0, 1)
    
    def __plot(self):
        """
        Plot the data points, projection points, perpendicular lines, and projection line.
        """
        # data points
        plt.scatter(self.X1[:, 0], self.X1[:, 1], label='positive', marker='o')
        plt.scatter(self.X0[:, 0], self.X0[:, 1], label='negative', marker='x')
        
        # projection points
        proj0 = self.X0.dot(self.w)
        proj1 = self.X1.dot(self.w)
        plt.scatter(self.w[0] * proj1, self.w[1] * proj1, color='g', label='projection of positive', marker='o')
        plt.scatter(self.w[0] * proj0, self.w[1] * proj0, color='r', label='projection of negative', marker='x')
        
        # perpendicular lines
        for i in range(len(proj1)):
            plt.plot([self.X1[i, 0], self.w[0] * proj1[i]], [self.X1[i, 1], self.w[1] * proj1[i]], 'g--', lw=0.4)
        for i in range(len(proj0)):
            plt.plot([self.X0[i, 0], self.w[0] * proj0[i]], [self.X0[i, 1], self.w[1] * proj0[i]], 'r--', lw=0.4)
            
        # center of each class
        plt.scatter(self.mu1[0], self.mu1[1], label='$\mu_1$', marker='^')
        plt.scatter(self.mu0[0], self.mu0[1], label='$\mu_0$', marker='s')
        
        # set x- and y-axis limits
        # concatenate all x and y coordinates of data points and projection points
        points_x = np.concatenate((self.X0[:, 0], self.X1[:, 0], self.w[0] * proj0, self.w[0] * proj1), axis=0)
        points_y = np.concatenate((self.X0[:, 1], self.X1[:, 1], self.w[1] * proj0, self.w[1] * proj1), axis=0)
        x_min, x_max = points_x.min(), points_x.max()
        y_min, y_max = points_y.min(), points_y.max()
        plt.axis('equal')  # equal scaling is necessary for perpendicular lines
        plt.xlim(x_min - 0.05, x_max + 0.05)
        plt.ylim(y_min - 0.05, y_max + 0.05)
        
        # projection line
        slope = self.w[1] / self.w[0]
        line_x = np.array(plt.xlim())
        line_y = slope * line_x
        plt.plot(line_x, line_y, color='black', lw=1)
        
        # decision boundary
        boundary_point = (self.mu0 + self.mu1) / 2
        boundary_slope = -1 / slope
        boundary_line_x = np.array(plt.xlim())
        boundary_line_y = boundary_slope * (boundary_line_x - boundary_point[0]) + boundary_point[1]
        plt.plot(boundary_line_x, boundary_line_y, color='blueviolet', lw=2, ls='-.', label='decision boundary')
        
        plt.title('LDA')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.legend()
        plt.show()
        
    @property
    def mu0(self) -> np.ndarray:
        return self.X0.mean(0)
    
    @property
    def mu1(self) -> np.ndarray:
        return self.X1.mean(0)


if __name__ == "__main__":
    model = LDA()

    data = np.array([
        [0.666, 0.091, 1],
        [0.243, 0.267, 1],
        [0.244, 0.056, 1],
        [0.342, 0.098, 1],
        [0.638, 0.16,  1],
        [0.656, 0.197, 1],
        [0.359, 0.369, 1],
        [0.592, 0.041, 1],
        [0.718, 0.102, 1],
        [0.697, 0.46,  0],
        [0.774, 0.376, 0],
        [0.633, 0.263, 0],
        [0.607, 0.317, 0],
        [0.555, 0.214, 0],
        [0.402, 0.236, 0],
        [0.481, 0.149, 0],
        [0.436, 0.21,  0],
        [0.557, 0.216, 0]])

    X = data[:, :-1]  # attributes
    y = data[:, -1]   # labels

    model.fit(X, y)
