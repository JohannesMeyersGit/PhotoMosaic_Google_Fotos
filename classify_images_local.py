import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models.efficientnet import efficientnet_b4, EfficientNet_B4_Weights


from PIL import Image
import json
import os
import requests
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob

# use graphics card if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the model following https://pytorch.org/vision/stable/models.html

# model = models.resnet18(pretrained=True)
weights = EfficientNet_B4_Weights.IMAGENET1K_V1


model = efficientnet_b4(weights=weights) # load the model with imagenet weights from the efficientnet library
model.eval()

model = model.to(device)

model.eval()




# load image from cached images and classify it with the model

all_ims = glob.glob('./cache/images/*.jpg')
No_of_ims = len(all_ims)

rand_idx = np.random.randint(0, No_of_ims)

test_im = all_ims[rand_idx]

im = Image.open(test_im)

im = im.resize((224, 224)) # resize to 224x224 pixels to fit the model input

im = transforms.ToTensor()(im).unsqueeze(0) # convert to tensor and add batch dimension


# classify the image using the model

with torch.no_grad():
    im = im.to(device)
    output = model(im)
    # get model output from the last convolutional layer
    output2 = output.squeeze().cpu()
    
    
# get the class labels from the imagenet classes from  https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json 3.5k
with open('imagenet_class_index.json') as f:
    classes = json.load(f)



# get the class label with the highest probability
class_idx = torch.argmax(output).item()
class_label = classes[str(class_idx)]

class_prob = torch.nn.functional.softmax(output, dim=1)[0, class_idx].item()

# get top 5 class labels and probabilities 
top5_prob, top5_idx = torch.topk(output, 5)
top5_prob = torch.nn.functional.softmax(top5_prob, dim=1).squeeze().tolist()
top5_idx = top5_idx.squeeze().tolist()

top5_labels = [classes[str(idx)] for idx in top5_idx]

print(top5_labels)
print(top5_prob)


print(class_label)


# show the image and the class label as title
plt.figure()
plt.title(class_label[1])
plt.imshow(plt.imread(test_im))
plt.show()




