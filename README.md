# UniVL

## version 0.2
在visual encoder之后sampling，并sampling了8个clip，每个clip长度为5 seconds \
在v0.2.5的时候发现，在encoder之前sampling结果只有原来的一半
## version 0.3
经过v0.2的经验，就把sampling环节定格在encoder之后 \
同时在modeling模块增加了self.clip_num,self.frame_num，改起来更加方便