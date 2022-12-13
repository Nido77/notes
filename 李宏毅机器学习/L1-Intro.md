[TOC]



# 1. 机器学习到底在干嘛

拟合一个万能函数来解决实际问题,这个函数的input可以是各种东西,输出也可以是各种东西![db71c8722701d9777a7d46e3af25d2d](https://cdn.jsdelivr.net/gh/Nido77/notesimage@main/img/db71c8722701d9777a7d46e3af25d2d.png)

# 2. 现在的机器学习任务?

> 一共有三种哦,别想着只有regression & classification

1. regression -- 生成一个值
2. classification -- 分类问题(一般生成的是整数)
3. structured learning -- 产生有结构的物体: 图,文章等,让机器学会创造(黑暗大陆来的)

<img src="https://cdn.jsdelivr.net/gh/Nido77/notesimage@main/img/7148d3114bad9aecdbdea6015689d2e.png" alt="7148d3114bad9aecdbdea6015689d2e" style="zoom: 33%;" />

# 3. 预测YouTube观看人数

![7174cb3457bc494211f00ba2b1a092c](https://cdn.jsdelivr.net/gh/Nido77/notesimage@main/img/7174cb3457bc494211f00ba2b1a092c.png)

> 输入日期和观看人数,预测明天的观看人数

以下是机器学习的三步骤

## 3.1写出带有未知参数的函数

> 猜测一下预测人数的f长什么样

比如说$\hat{y}=b + wx_1$(model)

* $x_1$指的是前一天的观看人数(feature)
* $\hat{y}$是预测今天观看人数(prediction)
* $b(bias)$和$w(weight)$是未知参数
* 这个公式的得出基于domain knowledge 也就是经验

## 3.2 定义Loss

> Loss是一个函数,输入是b,w,也就是可以写成$L(b,w)$, 代表函数拟合训练集的好坏

$b = 500, k=0.5 \Rightarrow \hat{y} = 500+0.5x$,怎么衡量这个模型的好坏呢? 计算Loss $L$

那么怎么计算$L$呢? 从训练集中计算:

* 估测值跟真实值的差距:
* Mean Absolute Error(MAE): $\frac{1}{N}\sum_{i=1}^N|y-\hat{y}|$
* Mean Square Error(MSE):$\frac{1}{2N}\sum_{i=1}^N(\hat{y}-y)^2$

## 3.3 最优化参数

> $w^{*},b^{*}=arg \min_{w,b}L$,也就是使得L最小的$w$和$b$

使用**Gradient Descent**,为了简化$b=0$

* **随机**选取初始$w$的值,记为$w_0$
* 计算$\frac{\partial{L}}{ \partial{w}}|_{w=w_0}$
* 更新$w$,$w_{1}=w_{0}-\alpha\frac{\partial{L}}{ \partial{w}}|_{w=w_0}$, 这里的$\alpha$叫做learning rate, 是自己设定的参数(这种自己设定的参数又叫做hyperparameter)
* 一直更新,直到达到想要的条件(这个条件也是一个hyperparameter)

两个参数是一样的,就是求b的偏导

### Gradient Descent找到的是local minima怎么办?(TODO)

不用担心,在高纬度上没有这个问题,以后笔记会解释

## 3.4 非线性模型

> 线性模型预测结果不太行,我们来使用非线性模型

### 重新定义模型(函数)



<img src="https://cdn.jsdelivr.net/gh/Nido77/notesimage@main/img/image-20221120160511234.png" alt="image-20221120160511234" style="zoom:33%;" />

比如说你的数据图是一条折线,你不管怎么调整$w$和$b$都不可能拟合的很好,这种来自于model的限制叫做**model bias**,那么我们就需要更复杂的model,如何生成呢?

<img src="https://cdn.jsdelivr.net/gh/Nido77/notesimage@main/img/image-20221120201513276.png" alt="image-20221120201513276" style="zoom: 33%;" />

转折点相加就好了,那如果不是这种分段的呢?可以用极限的思想,取两个间隔很小的点进行拟合,如下图所示

<img src="https://cdn.jsdelivr.net/gh/Nido77/notesimage@main/img/image-20221120202726383.png" alt="image-20221120202726383" style="zoom:33%;" />

也就是说,对于任意函数,我们都可以用这个分段函数进行拟合

那么又有一个问题了,我们怎么写出这个**蓝色函数的式子呢**?

==答案是==$\rm{sigmoid}:y=c\frac{1}{1+e^{-(b+wx_1)}}=c*sigmoid(b+wx_1)$, 蓝色的叫做hard sigmoid,通过调整$c,b,w$来进行拟合

<img src="https://cdn.jsdelivr.net/gh/Nido77/notesimage@main/img/image-20221120203831079.png" alt="image-20221120203831079" style="zoom: 33%;" />

那么回到之前的问题,红色线段怎么表示?如下图所示

<img src="https://cdn.jsdelivr.net/gh/Nido77/notesimage@main/img/%E6%8A%98%E7%BA%BF%E5%9B%BE1.png" alt="折线图1" style="zoom:33%;" />

1. $y = c_1*sigmoid(b_1+ w_1x_1)$
1. $y = c_2*sigmoid(b_2+ w_2x_1)$
1. $y = c_3*sigmoid(b_3+ w_3x_1)$

$$
y = b + \sum_{i}c_isigmoid(b_i+w_ix_1)
$$

那么我们预测人数的式子就转变成
$$
\begin{aligned}
&y=b+\sum_j w_jx_j\quad (这个式子是说第j+1天的观看人数是和前j天线性相关)\\
&y=b+\sum_{i}c_i*sigmoid(b_i+\sum_{j}w_{ij}x_j)\quad (这个式子是说有i段,和前j天有关)
\end{aligned}
$$
**一共有$i$段, $i$段的函数有$j$维**

那么**sigmoid括号**里的东西可以写成这样

![未命名绘图-第 5 页.drawio](https://cdn.jsdelivr.net/gh/Nido77/notesimage@main/img/%E6%9C%AA%E5%91%BD%E5%90%8D%E7%BB%98%E5%9B%BE-%E7%AC%AC%205%20%E9%A1%B5.drawio.svg)

那么可以化简成矩阵的样子

$$
\begin{bmatrix}
 r_1\\
 r_2\\
 r_3\\
 \end{bmatrix} =
 \begin{bmatrix}
 b_1\\
 b_2\\
 b_3\\
 \end{bmatrix}
 +
 \begin{bmatrix}
 w_{11}& w_{12}& w_{13}\\
 w_{21}& w_{22}& w_{23}\\
 w_{31}& w_{32}& w_{33}\\
 \end{bmatrix}
  \begin{bmatrix}
 x_1\\
 x_2\\
 x_3\\
 \end{bmatrix}
$$

$$
r=b+Wx
$$

之后通过sigmoid
$$
\begin{aligned}
&a_1= \frac{1}{1+e^{-r_1}}\\
&a=sigmoid(r)=\sigma(r)
\end{aligned}
$$
最后得到输出值
$$
y = b + c^Ta
$$


总结一下就是这样子的

<img src="https://cdn.jsdelivr.net/gh/Nido77/notesimage@main/img/image-20221121090206188.png" alt="image-20221121090206188" style="zoom: 33%;" />

再统一下notation, 将所有参数都叫做$\theta$

<img src="https://cdn.jsdelivr.net/gh/Nido77/notesimage@main/img/image-20221121091156915.png" alt="image-20221121091156915" style="zoom:33%;" />

### 定义Loss,并从训练数据中计算Loss

可以使用之前的MAE和MSE

### 对参数进行优化

>  $\theta^*=argmin_{\theta}L$

1. **随机选择**一个初始值$\theta_0$
2. 计算梯度$$g=\begin{bmatrix}\frac{\partial L}{\partial\theta_1}|_{\theta=\theta^0}\\ \frac{\partial L}{\partial\theta_2}|_{\theta=\theta^0} \\... \end{bmatrix}= \bigtriangledown L(\theta^0)$$
3. 更新参数$\theta_1=\theta_0-\alpha*g$

### 激活函数的选择

> 之前的sigmoid叫做激活函数,后面提到的relu也是

怎么表示之前说过的hard sigmoid呢,可以用两个relu叠起来

$relu=max(0,z)$

<img src="https://cdn.jsdelivr.net/gh/Nido77/notesimage@main/img/image-20221121093339855.png" alt="image-20221121093339855" style="zoom:33%;" />

大部分情况是relu比较好(Ng也这么说过)

### 增加网络深度以提高拟合程度

<img src="https://cdn.jsdelivr.net/gh/Nido77/notesimage@main/img/image-20221121093817640.png" alt="image-20221121093817640" style="zoom:33%;" />

这就是两层网络,就是把$a$作为输入值输入到下一层中,值得注意的是$W$和$W^{'}$不一样

## 3.5 什么叫Deep learning

> 拟合的过程就叫做神经网络,激活函数叫做神经元

<img src="https://cdn.jsdelivr.net/gh/Nido77/notesimage@main/img/image-20221121094726774.png" alt="image-20221121094726774" style="zoom:33%;" />

一些网络举例

<img src="https://cdn.jsdelivr.net/gh/Nido77/notesimage@main/img/image-20221121095232297.png" alt="image-20221121095232297" style="zoom:33%;" />

### 为什么更喜欢深的网络不喜欢死肥宅网络?(TODO)



