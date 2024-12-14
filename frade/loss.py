import torch
import torch.nn as nn
import torch.nn.functional as F

def contrastive_loss_AudioVideo(video, audio, targets, beta= 0.2):
    #audio, video: [b, embed_dims]
    cos_similarities = F.cosine_similarity(video, audio)
    b = cos_similarities.shape
    #print(cos_similarities)
    
    loss = torch.mean(
        targets.float()* cos_similarities + (1- targets.float())* torch.clamp((1- cos_similarities), min= beta)
    )
    
    return loss


class CenterLoss(nn.Module):
    def __init__(self, input_dims= 768):
        super().__init__()
        
        self.center = nn.Parameter(torch.randn(input_dims))
    
    def forward(self, feats, targets):
        real_feats = feats[targets == 0]
        if real_feats.shape[0] == 0:
            return 0.
        
        loss = F.mse_loss(real_feats, repeat(self.center, 'd -> b d', b= real_feats.shape[0]))
        return loss