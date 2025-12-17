import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import GUI

screen_width = 600
screen_height = 600

####################################

res = 8
with Image.open("Images/Seth Dutter-modified-min (1).png").resize((res,res)) as img:
    img_array = np.array(img)

display = GUI.Display(screen_height,screen_width)
display.img_board_setup(img_array)

display.pause()

#####################################

res = 32
with Image.open("Images/Seth Dutter-modified-min (1).png").resize((res,res)) as img:
    img_array = np.array(img)

display.img_board_setup(img_array)

display.pause()

#####################################

res = 128
with Image.open("Images/Seth Dutter-modified-min (1).png").resize((res,res)) as img:
    img_array = np.array(img)

display.img_board_setup(img_array)

display.pause()
display.close()

# print(type(img_array))
# print(img_array)
# print(np.round(img_array, 0))
#
# plt.imshow(img_array, cmap='gray')
# #plt.imshow(np.round(img_array, 0), cmap='gray')
# plt.show()