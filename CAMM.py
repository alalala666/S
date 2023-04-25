import torch
from torchvision import models
from torch.nn import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class CAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradient = None
        
    def hook(self, module, input, output):
        self.feature_map = output.detach()
        
    def backward_hook(self, module, grad_in, grad_out):
        self.gradient = grad_out[0].detach()
        
    def get_gradient(self):
        return self.gradient
        
    def get_activation(self, x):
        return self.feature_map
        
    def generate_CAM(self, image):
        self.model.eval()
        with torch.no_grad():
            output = self.model(image)
            pred_class = torch.argmax(output, dim=1).item()
            score = F.softmax(output, dim=1)[:, pred_class].item()
            self.model._modules.get(self.target_layer).register_forward_hook(self.hook)
            self.model._modules.get(self.target_layer).register_backward_hook(self.backward_hook)
            output.backward(torch.tensor([[1.0, 0.0, 0.0]]).T, retain_graph=True)
            gradient = self.get_gradient()
            activation = self.get_activation(image)
            weights = F.adaptive_avg_pool2d(gradient, 1)
            cam = torch.sum(weights * activation, dim=1).squeeze()
            cam = F.relu(cam)
            cam = F.interpolate(cam.unsqueeze(0), size=image.shape[2:], mode='bilinear', align_corners=False)
            cam = cam.squeeze().numpy()
            cam = np.maximum(cam, 0)
            cam = cam / cam.max() if cam.max() != 0 else cam
            return cam, pred_class, score
        
if __name__ == '__main__':
    # 載入模型
    model = models.resnet18(pretrained=True)
    # 設置CAM
    cam = CAM(model, 'layer4')
    # 載入圖像
    img_path = 'query_data/S/NonS/NonS_240.jpg'
    img = Image.open(img_path)
    img_tensor = F.to_tensor(img).unsqueeze(0)
    # 生成CAM
    cam_map, pred_class, score = cam.generate_CAM(img_tensor)
    # 顯示圖像和CAM
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[1].imshow(cam_map, cmap='jet')
    ax[1].axis('off')
    ax[1].set_title(f'Class: {pred_class}, Score: {score:.3f}')
    plt.show()
