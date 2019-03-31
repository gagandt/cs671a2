from PIL import Image, ImageDraw
import numpy as np
import os
import math

class img:
    label = ""
    length = 7
    width = 1
    color = [255,0,0]
    number = 1
    angle = 0
    
    def __init__(self, lab, leg, wid, ang, col):
        self.label = lab
        self.length = leg
        self.width = wid
        self.color = col
        self.angle = ang
    
    def create(self):
        if not os.path.exists(self.label):
            os.makedirs(self.label)
        
        os.chdir(self.label)
        
        self.label += "_"
            
        itr = 1
        print("N")

        while itr < 1001:
            
            c11 = 0
            while (c11 <= 28 and itr < 1001):
                c12 = 0
                while (c12 <= 28 and itr < 1001):
                    c21 = c11 + round(self.length * math.cos(self.angle))
                    c22 = c12 + round(self.length * math.sin(self.angle))
                    
                    
                    
                    #if (((c11 - c21) ** 2 + (c12 - c22) ** 2) < (self.length ** 2 - 5)):
                    if (c21 > 28 or c22 > 28 or c21 < 0 or c22 < 0):
                        break
                    else:
                        w, h = 28, 28
                        data = np.zeros((h, w, 3), dtype=np.uint8)
                        im = Image.fromarray(data, 'RGB')
                        #img.save('base.png')
                        
                        #im = Image.open("base.png")
                        d = ImageDraw.Draw(im)
            
                        cor1 = (c11, c12)
                        cor2 = (c21, c22)
                        line_color = (self.color[0], self.color[1], self.color[2])
                        d.line([cor1, cor2], fill=line_color, width=self.width)
                        im.save(self.label + str(itr)+".png")
                        print(self.label + str(itr)+".jpg")
                        
                        itr += 1
                        c12 += 1
                
                c11 += 1
        
        os.chdir("..")
                        
                        
