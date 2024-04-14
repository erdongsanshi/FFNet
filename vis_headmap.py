import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from Networks.FFNet import FFNet
import torch
from torchvision import transforms
import numpy as np
import torchvision
from PIL import Image
import cv2
import os
import torch.nn.functional as F

save_path = '/home/deeplearn/JupyterlabRoot/erdongsanshi/FFNet/'

img_dir = "/datasets/shanghaitech/part_A_final/test_data/images"

images=os.listdir(img_dir)
print(len(images))
model_path = '/home/deeplearn/JupyterlabRoot/erdongsanshi/FFNet/SHA_model.pth'
model = FFNet()
model.load_state_dict(torch.load(model_path))
model = model.eval()

input_H, input_W = 256,256

heatmap = np.zeros([input_H, input_W])

layer = model.ccsm1
print(layer)


def farward_hook(module, data_input, data_output):
    fmap_block.append(data_output)
    input_block.append(data_input)
     
i = 0
for img in images[:40]:
    print("第{}张图片".format(i+1))
    read_img = os.path.join(img_dir,img)
    image = Image.open(read_img).convert('RGB')
    
    image = image.resize((input_H, input_W))
    image = np.float32(image) / 255
    input_tensor = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])(image)
    
    input_tensor = input_tensor.unsqueeze(0)
    
    if torch.cuda.is_available():
        model = model.cuda()
        input_tensor = input_tensor.cuda()
        
    input_tensor.requires_grad = True
    fmap_block = list()
    input_block = list()

    layer.register_forward_hook(farward_hook)
    
    output,_ = model(input_tensor)
    _, _, h1, w1 = output.size()
    output = F.interpolate(output, size=(h1 * 8, w1 * 8), mode='bilinear', align_corners=True) / 64

    feature_map = fmap_block[0].mean(dim=1,keepdim=False).squeeze()
    
    feature_map[(feature_map.shape[0]//2-1)][(feature_map.shape[1]//2-1)].backward(retain_graph=True)

    grad = torch.abs(input_tensor.grad)

    grad = grad.mean(dim=1,keepdim=False).squeeze()
    
    heatmap = heatmap + grad.cpu().numpy()
    i += 1
    
    
cam = heatmap

cam = cam / cam.max()

cam = cv2.applyColorMap(np.uint8(cam*255), cv2.COLORMAP_JET)

cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
Image.fromarray(cam)
cv2.imwrite("%s/receptive_field1.png" % save_path, cam)
