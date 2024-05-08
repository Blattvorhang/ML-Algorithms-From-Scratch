# Decision Tree
## Question
Try using [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) to train an SVM on the Watermelon Dataset $3.0α$ using a 2nd degree polynomial kernel $K(x_i, x_j)=(\gamma \cdot x_i\cdot x_j)^2$ with different $\gamma$ parameters. Compare the differences in their support vectors, and finally recommend a suitable $\gamma$ parameter and explain the reason.

试使用[LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)，在西瓜数据集 $3.0α$ 上分别用不同 $\gamma$ 参数的2次多项式核 $K(x_i, x_j)=(\gamma \cdot x_i \cdot x_j)^2$ 训练一个SVM，比较他们支持向量的差别，最后推荐一个合适的 $\gamma$ 参数并说明理由。

![](./data.jpg)

## Answer
![](./scatter.png)
![](./gamma_5.png)
![](./gamma_20.png)
![](./gamma_35.png)
![](./gamma_50.png)

## Appendix
This subproject is based on the [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) library. To install the library, you can use the following command:

```bash
pip install -U libsvm-official
```

Then you can import the library in your code:

```python
from libsvm.svmutil import *
```