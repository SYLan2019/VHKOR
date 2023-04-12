from PIL import ImageEnhance
import os
import numpy as np
from PIL import Image


def brightnessEnhancement(root_path, img_name, brightness):#亮度增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_bri = ImageEnhance.Brightness(image)
    # brightness = 1.1+0.4*np.random.random()
    #brightness = 1.5
    image_brightened = enh_bri.enhance(brightness)
    return image_brightened


def contrastEnhancement(root_path, img_name, contrast):  # 对比度增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_con = ImageEnhance.Contrast(image)
    # contrast = 1.1+0.4*np.random.random()
    #contrast = 1.5
    image_contrasted = enh_con.enhance(contrast)
    return image_contrasted

def rotation(root_path, img_name, angle):
    img = Image.open(os.path.join(root_path, img_name))
    #random_angle = np.random.randint(-2, 2)*90
    # if random_angle==0:
    #  rotation_img = img.rotate(-90)
    # else:
    #     rotation_img = img.rotate( random_angle)
    # rotation_img.save(os.path.join(root_path,img_name.split('.')[0] + '_rotation.jpg'))
    rotation_img = img.rotate(angle)
    return rotation_img

def flip(root_path,img_name,t):   #翻转图像
    img = Image.open(os.path.join(root_path, img_name))
    if (t == 0):
        filp_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        filp_img = img.transpose(Image.FLIP_TOP_BOTTOM)
    # filp_img.save(os.path.join(root_path,img_name.split('.')[0] + '_flip.jpg'))
    return filp_img


def createImage(imageDir, saveDir):
    tag = []
    for name in os.listdir(imageDir):
        tag.append(int(name.split("_")[1]))
        if len(tag) == 1:
            i = 4
        elif len(tag) == 2:
            i = 19
        elif len(tag) == 3:
            i = 34
            tag = []
        name0 = name.split(".")[0][:-1]
        saveName = name0 + str(i) + ".jpg"
        saveImage = contrastEnhancement(imageDir, name, 1.1)
        saveImage.save(os.path.join(saveDir,saveName))
        i = i + 1
        saveName4 = name0 + str(i) + ".jpg"
        saveImage = contrastEnhancement(imageDir, name, 1.2)
        saveImage.save(os.path.join(saveDir, saveName4))
        i = i + 1
        saveName4 = name0 + str(i) + ".jpg"
        saveImage = contrastEnhancement(imageDir, name, 1.3)
        saveImage.save(os.path.join(saveDir, saveName4))
        i = i + 1
        saveName4 = name0 + str(i) + ".jpg"
        saveImage = contrastEnhancement(imageDir, name, 0.7)
        saveImage.save(os.path.join(saveDir, saveName4))
        i = i + 1
        saveName4 = name0 + str(i) + ".jpg"
        saveImage = contrastEnhancement(imageDir, name, 0.8)
        saveImage.save(os.path.join(saveDir, saveName4))
        i = i + 1
        saveName4 = name0 + str(i) + ".jpg"
        saveImage = contrastEnhancement(imageDir, name, 0.9)
        saveImage.save(os.path.join(saveDir, saveName4))
        i = i + 1
        saveName6 = name0 + str(i) + ".jpg"
        saveImage = brightnessEnhancement(imageDir, name, 1.1)
        saveImage.save(os.path.join(saveDir, saveName6))
        i = i + 1
        saveName2 = name0 + str(i) + ".jpg"
        saveImage2 = brightnessEnhancement(imageDir, name, 1.2)
        saveImage2.save(os.path.join(saveDir, saveName2))
        i = i + 1
        saveName2 = name0 + str(i) + ".jpg"
        saveImage2 = brightnessEnhancement(imageDir, name, 1.3)
        saveImage2.save(os.path.join(saveDir, saveName2))
        i = i + 1
        saveName2 = name0 + str(i) + ".jpg"
        saveImage2 = brightnessEnhancement(imageDir, name, 0.7)
        saveImage2.save(os.path.join(saveDir, saveName2))
        i = i + 1
        saveName2 = name0 + str(i) + ".jpg"
        saveImage2 = brightnessEnhancement(imageDir, name, 0.8)
        saveImage2.save(os.path.join(saveDir, saveName2))
        i = i + 1
        saveName2 = name0 + str(i) + ".jpg"
        saveImage2 = brightnessEnhancement(imageDir, name, 0.9)
        saveImage2.save(os.path.join(saveDir, saveName2))
        i = i + 1
        saveName1 = name0 + str(i) + ".jpg"
        saveImage1 = flip(imageDir, name, 0)
        saveImage1.save(os.path.join(saveDir, saveName1))
        i = i + 1
        saveName3 = name0 + str(i) + ".jpg"
        saveImage = flip(imageDir, name, 1)
        saveImage.save(os.path.join(saveDir, saveName3))
        i = i + 1
        saveName5 = name0 + str(i) + ".jpg"
        saveImage = rotation(imageDir, name, 180)
        saveImage.save(os.path.join(saveDir, saveName5))

imageDir1 = r"F:\AU data\data\outputdata\visual"
saveDir = r"F:\AU data\data\outputdata\visual1"
createImage(imageDir1, saveDir)
for name in os.listdir(imageDir1):
    tag = int((name.split("_")[2]).split(".")[0])
    if tag == 2:
        name0 = name.split(".")[0][:-1]
        i = 49
        saveName = name0 + str(i) + ".jpg"
        saveImage = brightnessEnhancement(imageDir1, name, 1.15)
        saveImage.save(os.path.join(saveDir, saveName))
        i = 50
        saveName = name0 + str(i) + ".jpg"
        saveImage = brightnessEnhancement(imageDir1, name, 0.85)
        saveImage.save(os.path.join(saveDir, saveName))
