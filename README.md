# AI-Challenger-Plant-Disease-Recognition
## 农作物病害检测
### 赛题简介
病虫害的诊断对于农业生产来说至关重要。本次农作物病虫害识别比赛邀请参赛者设计算法与模型，对图像中的农作物叶子进行病虫害识别。
### 数据说明
数据集有61个分类（按“物种-病害-程度”分），10个物种，27种病害（其中24个病害有分一般和严重两种程度），10个健康分类，47637张图片。每张图包含一片农作物的叶子，叶子占据图片主要位置。数据集随机分为训练（70%）、验证（10%）、测试A（10%）与测试B（10%）四个子数据集。训练集和验证集包含图片和标注的json文件，json文件中包含每一张图片和对应的分类ID。


| Label ID | Label Name   |
|----------|--------------|
|   0     |   apple healthy（苹果健康）     |
|   1     |    Apple_Scab general（苹果黑星病一般）    |
|   2     |    Apple_Scab serious（苹果黑星病严重）    |
|   3     |    Apple Frogeye Spot（苹果灰斑病）    |
|   4     |    Cedar Apple Rust  general（苹果雪松锈病一般）  |
|   5     |    Cedar Apple Rust serious（苹果雪松锈病严重）    |
|   6     |    Cherry healthy（樱桃健康）    |
|   7     |    Cherry_Powdery Mildew  general（樱桃白粉病一般）    |
|   8     |    Cherry_Powdery Mildew  serious（樱桃白粉病严重）    |
|   9     |    Corn healthy（玉米健康）    |
|   10    |  Cercospora zeaemaydis Tehon and Daniels general（玉米灰斑病一般）|
|   11    |  Cercospora zeaemaydis Tehon and Daniels  serious（玉米灰斑病严重） |
|   12    |   Puccinia polysora  general（玉米锈病一般）     |
|   13    |    Puccinia polysora serious（玉米锈病严重）    |
|   14    |    Corn Curvularia leaf spot fungus general（玉米叶斑病一般）    |
|   15    |    Corn Curvularia leaf spot fungus  serious（玉米叶斑病严重）    |
|   16    |    Maize dwarf mosaic virus（玉米花叶病毒病）    |
|   17    |    Grape heathy（葡萄健康）    |
|   18     |   Grape Black Rot Fungus general（葡萄黑腐病一般）    |
|   19     |    Grape Black Rot Fungus serious（葡萄黑腐病严重）   |
|   20     |   Grape Black Measles Fungus general（葡萄轮斑病一般）    |
|   21     |   Grape Black Measles Fungus serious（葡萄轮斑病严重）    |
|   22     |    Grape Leaf Blight Fungus general（葡萄褐斑病一般）   |
|   23     |    Grape Leaf Blight Fungus  serious（葡萄褐斑病严重）   |
|   24     |    Citrus healthy（柑桔健康）   |
|   25     |    Citrus Greening June  general（柑桔黄龙病一般）   |
|   26     |    Citrus Greening June  serious（柑桔黄龙病严重）   |
|   27     |    Peach healthy（桃健康）   |
|   28     |    Peach_Bacterial Spot general（桃疮痂病一般）   |
|   29     |   Peach_Bacterial Spot  serious（桃疮痂病严重）    |
|   30     |   Pepper healthy（辣椒健康）    |
|   31     |   Pepper scab general（辣椒疮痂病一般）   |
|   32     |   Pepper scab  serious（辣椒疮痂病严重）    |
|   33     |    Potato healthy（马铃薯健康）   |
|   34     |    Potato_Early Blight Fungus general（马铃薯早疫病一般）   |
|   35     |   Potato_Early Blight Fungus serious（马铃薯早疫病严重）    |
|   36     |   Potato_Late Blight Fungus general（马铃薯晚疫病一般）    |
|   37     |    Potato_Late Blight Fungus  serious（马铃薯晚疫病严重）   |
|   38     |    Strawberry healthy（草莓健康）   |
|   39     |    Strawberry_Scorch general（草莓叶枯病一般）   |
|   40     |   Strawberry_Scorch serious（草莓叶枯病严重）    |
|   41     |    Tomato healthy（番茄健康）   |
|   42     |    tomato powdery mildew  general（番茄白粉病一般）   |
|   43     |    tomato powdery mildew  serious（番茄白粉病严重）   |
|   44     |    Tomato Bacterial Spot Bacteria general（番茄疮痂病一般）   |
|   45     |    Tomato Bacterial Spot Bacteria  serious（番茄疮痂病严重）   |
|   46     |    Tomato_Early Blight Fungus general（番茄早疫病一般）   |
|   47     |    Tomato_Early Blight Fungus  serious（番茄早疫病严重）   |
|   48     |    Tomato_Late Blight Water Mold  general（番茄晚疫病菌一般）   |
|   49     |    Tomato_Late Blight Water Mold serious（番茄晚疫病菌严重）   |
|   50     |    Tomato_Leaf Mold Fungus general（番茄叶霉病一般）   |
|   51     |    Tomato_Leaf Mold Fungus serious（番茄叶霉病严重）   |
|   52     |    Tomato Target Spot Bacteria  general（番茄斑点病一般）   |
|   53     |    Tomato Target Spot Bacteria  serious（番茄斑点病严重）   |
|   54     |     Tomato_Septoria Leaf Spot Fungus  general（番茄斑枯病一般）  |
|   55     |    Tomato_Septoria Leaf Spot Fungus  serious（番茄斑枯病严重）   |
|   56     |    Tomato Spider Mite Damage general（番茄红蜘蛛损伤一般）   |
|   57     |    Tomato Spider Mite Damage serious（番茄红蜘蛛损伤严重）   |
|   58     |    Tomato YLCV Virus general（番茄黄化曲叶病毒病一般）   |
|   59     |    Tomato YLCV Virus  serious（番茄黄化曲叶病毒病严重）  |
|   60     |     Tomato Tomv（番茄花叶病毒病）  |


### 结果提交说明
选手返回的结果应存为JSON文件，提交结果应包含照片id与所属的分类id，格式如下：
```json
[
	{
		"image_id": "72e0dfb8d1460203d90ce46bdc0a0fb7b84a665a.jpg",
        "disease_class":1
    },
    ...
]
```

### 算法思路
* 迁移学习，使用Inception-v3，准确率可以达到75%
* 学习率指数衰减
  decayed_lr = lr \cdot decay_rate^{step/decay_steps}

* 数据读取
  ```Python
  # 读取原始图像数据
  image_row_data = tf.gfile.FastGFile(image_dir, 'r').read()
  # jpeg解码图像，得到三维矩阵，可用于显示图像
  img_data = tf.image.decode_jpeg(image_row_data)
  # 将数据类型转化为实数，以便程序处理
  img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
  ```
* 数据输入
  Inception-v3的输入层--Cast节点可以接受任意尺寸的矩阵输入，然后再扩展维度为 (1，？，？，3）的图片矩阵，最后统一resize为（1,229,229,3）的图片矩阵，inception只接受单张图片的输入，不接受batch图片;

* 最后一层千万不能用Relu

### 遇到的问题
* unicode编码
* 模型怎么选择
* xgboost

### 优化思路
* 动态指数衰减lr 已完成
* 加几层conv、fc 已完成
* data argument 已完成
* 集成多种模型xgboost

### 版本说明
v1--使用inception-v3瓶颈层的结果进行分类，准确率可达65%
v2--加了使用test部分测试的代码，写入json，可供提交
v3--增加了学习率指数衰减，模型保存，每1000步计算一次在validiation的正确率，增加了三层全连接层加强分类效果(效果不是很好),增加将结果保存为json的代码
v4--增加了数据增强，尚未实验

### Submit
test_result-v1.json: 仅是为了测试格式是否正确
test_result-v2.json： 使用Inception-v3训练了10000步，对应V3的代码，准确率75.459% 220
