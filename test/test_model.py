from models.yolo11_star_seg import StarYOLOSegModel
import torch

model = StarYOLOSegModel(num_classes=2).cuda()
x = torch.randn(1, 3, 640, 640).cuda()
out = model(x)

print("Final segmentation mask logits:", out.shape)
