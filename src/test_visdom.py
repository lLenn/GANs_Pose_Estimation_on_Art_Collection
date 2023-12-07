import visdom
import numpy as np

vis = visdom.Visdom(env="test")
for i in range(10):
    vis.line(
        Y=np.ones((1,))*(i%2),
        X=np.ones((1))*(i+10),
        win=str(5),
        update='append'
    )