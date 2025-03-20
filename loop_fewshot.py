import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.models
#!pip3 install scipy
import torchvision.transforms as transforms
from easyfsl.datasets import CUB
def set_seeds(seed):
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.mps.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
batch_size = 32
num_workers = 1

# Ensure that you're working from the root of the repository
# and that CUB's images are in ./data/CUB/images/
transform = transforms.Compose(
    [transforms.ToTensor(),
    #transforms.Normalize((0.5), (0.5))]) # for mnist (1-channel)
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop((224, 224)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#testset = torchvision.datasets.ImageFolder(root=root, transform=transform)
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
#import timm
#net = timm.create_model('aimv2_large_patch14_224.apple_pt')
datasets = []
fn1_s = []
fn2_s = []
root = "/root/.cache/kagglehub/datasets/wenewone/cub2002011/versions/7/CUB_200_2011/images"
testset_0 = torchvision.datasets.ImageFolder(root=root, transform=transform)
datasets.append(testset_0)
fn1_s.append("vit_s_14_cub_embeddings.pt")
fn2_s.append("cub_class_labels.pt")
root = "/root/.cache/kagglehub/datasets/arjunashok33/miniimagenet/versions/1"
testset_1 = torchvision.datasets.ImageFolder(root=root, transform=transform)
datasets.append(testset_1)
fn1_s.append("vit_s_14_miniimagenet_embeddings.pt")
fn2_s.append("mini_imagenet_class_labels.pt")
testset_2 = torchvision.datasets.FGVCAircraft(root='./data', split="test",
                                    download=True, transform=transform)
datasets.append(testset_2)
fn1_s.append("vit_s_14_aircraft_embeddings.pt")
fn2_s.append("aircraft_class_labels.pt")
testset_3 = torchvision.datasets.CIFAR100(root='./data', train=False,
                                    download=True, transform=transform)
datasets.append(testset_3)
fn1_s.append("vit_s_14_cifar_embeddings.pt")
fn2_s.append("cifar_class_labels.pt")
testset_4 = torchvision.datasets.Flowers102(root='./data', split="test",
                                    download=True, transform=transform)
datasets.append(testset_4)
fn1_s.append("vit_s_14_flowers_embeddings.pt")
fn2_s.append("flowers_class_labels.pt")
'''
from transformers import CLIPModel
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
import torch.nn as nn
import torch
device = torch.device('cuda')
class CLIPWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model.to(device)
    def forward(self, x):
        x = self.base_model.get_image_features(x)
        return x
model = CLIPWrapper(model)
'''
# Remove the classification head: we want embeddings, not ImageNet predictions
#model.head = nn.Flatten()

# If you have a GPU, use it!
for item in range(len(datasets)):
    set_seeds(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    dataloader = torch.utils.data.DataLoader(datasets[item], batch_size=batch_size,
                                            shuffle=False, num_workers=1)
    from easyfsl.utils import predict_embeddings
    
    embeddings_df = predict_embeddings(dataloader, model, device=device, fn1=fn1_s[item], fn2=fn2_s[item])
    
    print(embeddings_df)
    from easyfsl.methods import PrototypicalNetworks, Finetune
    from easyfsl.datasets import FeaturesDataset
    
    features_dataset = FeaturesDataset.from_dataframe(embeddings_df)
    
    # Default backbone if we don't specify anything is Identity.
    # But we specify it anyway for clarity and robustness.
    few_shot_classifier = PrototypicalNetworks(backbone=nn.Identity(), feature_normalization=True)
    #few_shot_classifier = Finetune(backbone=nn.Identity(), feature_normalization=True)
    from easyfsl.samplers import TaskSampler
    task_sampler = TaskSampler(
        features_dataset,
        n_way=5,
        n_shot=5,
        n_query=15,
        n_tasks=1000, # test on more than 1000 tasks (like 100000)
    )
    features_loader = DataLoader(
        features_dataset,
        batch_sampler=task_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=task_sampler.episodic_collate_fn,
    )
    from easyfsl.utils import evaluate
    
    accuracy = evaluate(
        few_shot_classifier,
        features_loader,
        device="cpu",
    )
    print(f"Average accuracy : {(100 * accuracy):.2f} %") # aircraft: exponent 0.6 (73.96%), exponent 1 (73.42%), exponent 0.8 (74.05%)
    # CIFAR-100 (5-way 5-shot): 96.30% with exponent 1, 96.33% with exponent 0.9
    # flowers102 (5-way 5-shot): 99.93% with exponent 1, 99.96% with exponent 0.9
    # 0.9 seems to outperform 1 within margin of error a lot of times...
    # try doing 1 shot tomorrow (and even 10-class or 20-class 1-shot, to see if the exponent somehow improves the approximation)
    # or maybe moving the prototypes a little bit so that they are a little farther away from each other