# AI-Challenger-Plant-Disease-Recognition
## 农作物病害检测
详情请见[CSDN](https://blog.csdn.net/qq_40859461/article/details/84199358#commentsedit)

## 环境配置
python==2.7

tensorflow==1.2.1

## 使用方法
* 更改 plot.py 脚本中路径，运行该脚本，可以绘出数据分布的直方图
* 下载预训练模型 [Inception-V3](https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip
)
* 更改 plant_disease.py 中的输入文件路径，输出文件路径，预训练模型文件路径
* 在 code 路径下直接运行 python plant_disease.py
* 训练完成后会直接使用训练得到的参数预测 testA 数据集，生成可以用来直接提交的 json 文件

## 大佬开源分享
* [spytensor/plants_disease_detection](https://github.com/spytensor/plants_disease_detection)
  * 框架：pytorch 
  * 最终成绩：0.875
  
* [bochuanwu/Agricultural-Disease-Classification](https://github.com/bochuanwu/Agricultural-Disease-Classification)
  * 框架：keras
  * 最终成绩：0.88658

## 其他
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
