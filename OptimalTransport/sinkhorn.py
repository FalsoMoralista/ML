import timm
import torch
import torch.nn.functional as F
from PIL import Image

def sinkhorn_knopp(Q, num_iters=3):
    Q = Q / Q.sum()  # make total mass = 1
    K, B = Q.shape
    for _ in range(num_iters):
        Q /= Q.sum(dim=1, keepdim=True)
        Q /= K
        Q /= Q.sum(dim=0, keepdim=True)
        Q /= B
    Q *= B  # ensure each column sums to 1
    return Q.T  # return shape (B, K)


if not torch.cuda.is_available():
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0') # this is overwritten by the init_distributed() function (which assigns the corresponding rank for each process)
    torch.cuda.set_device(device)    

pretrained_path = '/home/rtcalumby/adam/luciano/PlantCLEF2025/PlantCLEF2025/pretrained_models/vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all/model_best.pth.tar'

model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=False, num_classes=7806, checkpoint_path=pretrained_path)
model.head = torch.nn.Identity() # Replace classification head by identity layer for feature extraction
model = model.to(device)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

centroids = ['centroid_0.jpg', 'centroid_1.jpg', 'centroid_2.jpg'] # flower, hand, plant

query = ['0b6ab5c99e452b8f6c7ea96a25cba601fefd41d8.jpg'] # (flower) Should be closer to centroid 0


data_path = '../utils/1361520/{}'
#
C = [model(transforms(Image.open(data_path.format(img))).unsqueeze(0).to(device)) for img in centroids]
C = [torch.nn.functional.normalize(centroid, dim=1).squeeze(0) for centroid in C] # Normalize centroids
print('Centroid size:', torch.stack(C).size())
query_img_embedding =  model(transforms(Image.open(data_path.format(query[0]))).unsqueeze(0).to(device))
Q = torch.nn.functional.normalize(query_img_embedding, dim=1).squeeze(0)

similarities = [torch.nn.functional.cosine_similarity(Q.unsqueeze(0), centroid.unsqueeze(0)) for centroid in C]
print('Cosine Similarities', similarities, ' similarities size', torch.stack(similarities).size())

closest_sample = torch.argmax(torch.stack(similarities), dim=0)
print('Closest sample:', closest_sample.item())

epsilon = 1#5e-2
soft_assignments = F.softmax(torch.stack(similarities) / epsilon, dim=0)
print('Soft assignments:', soft_assignments)


logits = torch.matmul(Q.unsqueeze(0) , torch.stack(C).T) / epsilon
print('logits shape', logits.size())
Q = torch.exp(logits).T

assignments = sinkhorn_knopp(Q, 10)  # shape: (1, 3)
print('Online assignments:', assignments)
