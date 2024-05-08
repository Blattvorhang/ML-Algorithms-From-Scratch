import numpy as np
from libsvm.svmutil import svm_train, svm_predict
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import matplotlib
matplotlib.use("TkAgg")

# Chinese
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class SVMUI:
    def __init__(self, X, y) -> None:
        self.root = tk.Tk()
        self.root.title('SVM')
        
        self.X = X
        self.y = y
        
        self.gamma = 0
        self.model = None
        
        self.scale_gamma = tk.Scale(
            self.root,
            label='gamma',
            from_=0,
            to=50,
            resolution=5,
            orient=tk.HORIZONTAL,
            length=200,
            showvalue=True,
            command=self.updateGamma
        )
        self.scale_gamma.set(self.gamma)
        self.scale_gamma.grid(row=1, column=0, padx=10, pady=3)
        
        self.btn_train = tk.Button(
            self.root,
            text='训练',
            command=self.train,
            width=8
        )
        self.btn_train.grid(row=1, column=1, pady=10)
        
        self.create_figure()
        self.root.mainloop()
                
    def train(self) -> None:
        self.model = svm_train(self.y.tolist(), self.X.tolist(), self.param)
        self.plot_decision_boundary()
        
    def updateGamma(self, gamma) -> None:
        self.gamma = float(gamma)
        # self.train()
        
    def create_figure(self) -> None:
        self.figure = Figure((6, 6))
        self.draw = self.figure.add_subplot(111)
        
        x_min, x_max = self.X[:, 0].min() - 0.2, self.X[:, 0].max() + 0.2
        y_min, y_max = self.X[:, 1].min() - 0.2, self.X[:, 1].max() + 0.2
        self.draw.set_xlim(x_min, x_max)
        self.draw.set_ylim(y_min, y_max)
        self.draw.set_xlabel('密度')
        self.draw.set_ylabel('含糖率')
        self.draw.set_title('SVM Decision Boundary')
        if self.model is None:
            self.draw.scatter(self.X[:, 0], self.X[:, 1], c=self.y, marker='o', edgecolors='k')
            
        self.canvas_plot = FigureCanvasTkAgg(self.figure, self.root)
        self.canvas_plot.get_tk_widget().grid(row=0, column=0, columnspan=2)
        
    def plot_decision_boundary(self) -> None:
        self.create_figure()
        
        x_min, x_max = self.X[:, 0].min() - 0.2, self.X[:, 0].max() + 0.2
        y_min, y_max = self.X[:, 1].min() - 0.2, self.X[:, 1].max() + 0.2

        # Create a grid that covers all data points
        step_size = 0.01
        xx, yy = np.meshgrid(np.arange(x_min, x_max + step_size, step_size),
                             np.arange(y_min, y_max + step_size, step_size))
        grid = np.c_[xx.ravel(), yy.ravel()]

        # Predict the grid points
        _, _, dec = svm_predict([0]*len(grid), grid.tolist(), self.model, '-q')

        # Reshape the predicted values to match xx and yy
        Z = np.array(dec).reshape(xx.shape)
        Z[Z >= 0] = 1
        Z[Z < 0] = 0

        # Plot the decision boundary
        self.draw.contourf(xx, yy, Z, alpha=0.5)  # levels=[Z.min(), 0, Z.max()]
        self.draw.scatter(self.X[:, 0], self.X[:, 1], c=self.y, marker='o', edgecolors='k')
        self.canvas_plot.draw()
    
    @property
    def param(self) -> str:  # polynomial kernel with degree 2
        return f'-t 1 -d 2 -g {self.gamma}'


if __name__ == '__main__':
    data = np.array([
        [0.697, 0.460, 1],
        [0.774, 0.376, 1],
        [0.634, 0.264, 1],
        [0.608, 0.318, 1],
        [0.556, 0.215, 1],
        [0.403, 0.237, 1],
        [0.481, 0.149, 1],
        [0.437, 0.211, 1],
        
        [0.666, 0.091, 0],
        [0.243, 0.267, 0],
        [0.245, 0.057, 0],
        [0.343, 0.099, 0],
        [0.639, 0.161, 0],
        [0.657, 0.198, 0],
        [0.360, 0.370, 0],
        [0.593, 0.042, 0],
        [0.719, 0.103, 0]])

    X = data[:, :-1]  # attributes
    y = data[:, -1]   # labels
    
    SVMUI(X, y)
    