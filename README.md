# UniVL

## version 0.2
在visual encoder之后sampling，并sampling了8个clip，每个clip长度为5 seconds \
在v0.2.5的时候发现，在encoder之前sampling结果只有原来的一半

---
## version 0.3
经过v0.2的经验，就把sampling环节定格在encoder之后 \
同时在modeling模块增加了self.clip_num,self.frame_num，改起来更加方便

---
## version 0.4
### version 0.4.0
similarity在cross encoder之后进行计算，效果较差
### version 0.4.1
利用几个最佳实验设置选项，在3*8的情况下跑出最好结果

---
## version 0.5
### version 0.5.1
1. visual encoder使用了VideoSwinT，然而效果掉了一半。怀疑是patch的处理不太对劲，打算更换一个cross encoder
2. eval的时候发现，把visual mask全部置为1将非常有利于效果的提升。现在还没做baseline的实验
3. 怀疑eval的写法有些问题，当前还没改

### version 0.5.2
1. 结构精简为swin、cross、dec。其中dec来自bert.LMHead
2. 把预训练模型载入，具体改的部分在xxx_simple.py文件里，同时意识到之前eval的时候可能没有载入对应的weight


---
## version 0.6
### version 0.6.1
1. 实现了swin backbone+Transformer backbone的正常结果，同时将siamese sampling通过trigger的方式融入，但是由于资源等原因无法跑过SOTA


