import os
import numpy as np
from skimage import io

class data:
    
    def __init__(self, directory):
        
        os.chdir(directory)


        class_labels = []
        all_images = []
        
        label_var = 0
        WIDTH = HEIGHT = 28
        
        for i in range(2):
            for j in range(2):
                for k in range(12):
                    for l in range(2):
                        label = str(i) + "_" + str(j) + "_" + str(k) + "_" + str(l)
                        
                        os.chdir(label)
                        count = 1
                        
                        while (count < 1001):
                            img = io.imread(label + "_" + str(count) +".jpg" , as_grey=False)
                            img = img.reshape([WIDTH, HEIGHT, 3])
                            all_images.append(img)
                            
                            class_labels.append(label_var)
                            count += 1
                        
                        print(label)
                        os.chdir("..")
                        label_var += 1
        
        return (np.array(all_images) , np.array(class_labels))
        