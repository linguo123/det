# PaddleClas实现图像分类Baseline

通过目标检测进行安全帽检测的应用。

# 一、项目背景

自己在学人工智能，目标检测是人工智能重要的一个研究方向。安全帽的检测，是一个非常好的练手机会。

![](https://ai-studio-static-online.cdn.bcebos.com/5f2066c06b0746c09b13a58e7e015ccb5b5e3616d8414ea8b34ae40f248be77d)


# 二、数据集简介

VOC数据集默认的JPEGImages和Annotations

## 1.数据加载和预处理


```python
import paddlex as pdx
from paddlex import transforms as T

# 数据的加载和预处理
transform = T.Normalize(mean=[127.5], std=[127.5])

# 训练数据集
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)

# 评估数据集
eval_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)

print('训练集样本量: {}，验证集样本量: {}'.format(len(train_dataset), len(eval_dataset)))
```

训练集样本量: 60000，验证集样本量: 10000


## 2.数据集查看


```python
print('图片：')
print(type(train_dataset[0][0]))
print(train_dataset[0][0])
print('标签：')
print(type(train_dataset[0][1]))
print(train_dataset[0][1])

# 可视化展示
plt.figure()
plt.imshow(train_dataset[0][0].reshape([28,28]), cmap=plt.cm.binary)
plt.show()

```


# 三、模型选择和开发

详细说明你使用的算法。此处可细分，如下所示：

## 1.模型组网

![](https://ai-studio-static-online.cdn.bcebos.com/08542974fd1447a4af612a67f93adaba515dcb6723ff4484b526ff7daa088915)


```python
# 模型网络结构搭建
network = paddle.nn.Sequential(
    paddle.nn.Flatten(),           # 拉平，将 (28, 28) => (784)
    paddle.nn.Linear(784, 512),    # 隐层：线性变换层
    paddle.nn.ReLU(),              # 激活函数
    paddle.nn.Linear(512, 10)      # 输出层
)
```

## 2.模型网络结构可视化


```python
# 模型封装
model = paddle.Model(network)
```
## 模型可视化

![](https://ai-studio-static-online.cdn.bcebos.com/c202c103518b45ffb82c892bc37b8d8a2e6955bef56445f0b1a5bbb70c00e7d5)
![](https://ai-studio-static-online.cdn.bcebos.com/0905f47db9a14d06bf0599b227275585b8f43e9a50a242c9a3504248a61208f2)





## 3.模型训练


```python
# 配置优化器、损失函数、评估指标
model.prepare(paddle.optimizer.Adam(learning_rate=0.001, parameters=network.parameters()),
              paddle.nn.CrossEntropyLoss(),
              paddle.metric.Accuracy())
              
# 启动模型全流程训练
model.fit(train_dataset,  # 训练数据集
          eval_dataset,   # 评估数据集
          epochs=5,       # 训练的总轮次
          batch_size=64,  # 训练使用的批大小
          verbose=1)      # 日志展示形式
```
预测结果：
```
{'category_id': 0, 'category': 'head', 'bbox': [301.3816223144531, 187.12403869628906, 18.0198974609375, 23.79364013671875], 'score': 0.9641125202178955}

{'category_id': 0, 'category': 'head', 'bbox': [98.42637634277344, 186.95138549804688, 17.62054443359375, 22.250213623046875], 'score': 0.9461163282394409}

{'category_id': 0, 'category': 'head', 'bbox': [60.45779037475586, 186.9356231689453, 18.03213119506836, 24.173797607421875], 'score': 0.940744161605835}

{'category_id': 0, 'category': 'head', 'bbox': [244.22019958496094, 187.01390075683594, 16.083084106445312, 22.99420166015625], 'score': 0.940227210521698}

{'category_id': 0, 'category': 'head', 'bbox': [136.5784912109375, 189.411865234375, 18.008148193359375, 22.964599609375], 'score': 0.8962541222572327}

{'category_id': 0, 'category': 'head', 'bbox': [21.016231536865234, 183.6320343017578, 17.833927154541016, 23.470489501953125], 'score': 0.886038601398468}

{'category_id': 0, 'category': 'head', 'bbox': [176.54640197753906, 188.69781494140625, 17.38409423828125, 23.967681884765625], 'score': 0.8675879240036011}

{'category_id': 0, 'category': 'head', 'bbox': [337.97821044921875, 186.59945678710938, 16.95550537109375, 22.102203369140625], 'score': 0.8598908185958862}

{'category_id': 0, 'category': 'head', 'bbox': [267.6236267089844, 185.5501708984375, 16.89678955078125, 23.020751953125], 'score': 0.8243097066879272}

{'category_id': 0, 'category': 'head', 'bbox': [210.44265747070312, 189.28919982910156, 18.027557373046875, 23.715240478515625], 'score': 0.786725640296936}

{'category_id': 0, 'category': 'head', 'bbox': [376.5035400390625, 195.9794158935547, 14.513916015625, 20.0159912109375], 'score': 0.5946234464645386}

未佩戴安全帽人员总数为: 11
```
    


## 4.模型评估测试


```python
# 模型评估，根据prepare接口配置的loss和metric进行返回
result = model.evaluate(eval_dataset, verbose=1)

print(result)
```

![](https://ai-studio-static-online.cdn.bcebos.com/3aac06f9a535431098b3b57f4d055e343783a184af564feda8798d1066a2fbdf)  


## 5.模型预测

### 5.1 批量预测

使用model.predict接口来完成对大量数据集的批量预测。


```python
# 进行预测操作
result = model.predict(eval_dataset)

# 定义画图方法
def show_img(img, predict):
    plt.figure()
    plt.title('predict: {}'.format(predict))
    plt.imshow(img.reshape([28, 28]), cmap=plt.cm.binary)
    plt.show()

# 抽样展示
indexs = [2, 15, 38, 211]

for idx in indexs:
    show_img(eval_dataset[idx][0], np.argmax(result[0][idx]))
```

    Predict begin...
    step 10000/10000 [==============================] - 1ms/step        
    Predict samples: 10000


### 5.2 单张图片预测

采用model.predict_batch来进行单张或少量多张图片的预测。


```python
# 读取单张图片
image = eval_dataset[501][0]

# 单张图片预测
result = model.predict_batch([image])

# 可视化结果
show_img(image, np.argmax(result))
```

# 四、效果展示

项目效果展示以及运行请看https://aistudio.baidu.com/aistudio/projectdetail/2277102

# 五、总结与升华

这是人工智能学习的一小步，但是是一个质的飞跃

不管懂不懂，都可以先试一试没学到了就是赚到了。

# 个人简介

我在AI Studio上获得青铜等级，点亮1个徽章，来互关呀~ https://aistudio.baidu.com/aistudio/personalcenter/thirdview/543525


