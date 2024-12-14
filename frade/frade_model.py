import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
import copy

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
print(sys.path)

from timm import create_model
from timm.models.vision_transformer import Block
from utils.Block_with_Bias import Block_qkv_Bias


# 待加入crossattn
class QKV_Adapter_ViT_spatio_filter(nn.Module):
    def __init__(self, input_dims, nums_head= 12, reduction= 2):
        super().__init__()
        
        self.input_dims = input_dims
        self.nums_head = nums_head
        
        lpf_mask = torch.zeros((14,14))
        
        for x in range(14):
            for y in range(14):
                if ((x- (14-1)/2)**2 + (y-(14-1)/2)**2) < 3:
                    lpf_mask[y,x] = 1
        
        self.mask = lpf_mask
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels= input_dims, out_channels= input_dims// reduction, kernel_size= 1, bias= False),
            nn.GELU(),
            nn.Conv2d(in_channels= input_dims// reduction, out_channels= input_dims, kernel_size= 3, padding= 1, bias= False),
            nn.GELU()
        )
        
        self.q = nn.Conv2d(in_channels= input_dims, out_channels= input_dims, kernel_size= 1, bias= False)
        self.k = nn.Conv2d(in_channels= input_dims, out_channels= input_dims, kernel_size= 1, bias= False)
        self.v = nn.Conv2d(in_channels= input_dims, out_channels= input_dims, kernel_size= 1, bias= False)
    
    def filter(self, x, H, W):
        out = torch.fft.fftn(x, dim= (2,3))
        out = torch.roll(out, (H//2, W//2), dims= (2,3))
        out_low = out* self.mask.to(x.device)
        out = torch.abs(torch.fft.ifftn(out_low, dim= (2,3)))
        
        return out
        
    def forward(self, x):
        batchsize, n, dims = x.shape
        
        H = W = int(math.sqrt(n-1))
        images = x[:,1:,:].view(batchsize, H, W, -1).permute(0,3,1,2)
        
        out = self.conv(images)
        
        out = self.filter(out, H, W)
        
        q = self.q(out).view(batchsize, self.input_dims, -1).permute(0,2,1).contiguous().view(batchsize, n-1, self.nums_head, dims// self.nums_head).permute(0,2,1,3)
        k = self.k(out).view(batchsize, self.input_dims, -1).permute(0,2,1).contiguous().view(batchsize, n-1, self.nums_head, dims// self.nums_head).permute(0,2,1,3)
        v = self.v(out).view(batchsize, self.input_dims, -1).permute(0,2,1).contiguous().view(batchsize, n-1, self.nums_head, dims// self.nums_head).permute(0,2,1,3)
        
        return [q, k, v]

class QKV_Adapter_ViT_audio_filter(nn.Module):
    def __init__(self, input_dims, nums_head= 12, reduction= 2):
        super().__init__()
        
        self.input_dims = input_dims
        self.nums_head = nums_head
        
        lpf_mask = torch.zeros((4,5))
        
        for x in range(5):
            for y in range(4):
                if ((x- (5-1)/2)**2 + (y-(4-1)/2)**2) < 1:
                    lpf_mask[y,x] = 1
        
        self.mask = lpf_mask
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels= input_dims, out_channels= input_dims// reduction, kernel_size= 1, bias= False),
            nn.GELU(),
            nn.Conv2d(in_channels= input_dims// reduction, out_channels= input_dims, kernel_size= 3, padding= 1, bias= False),
            nn.GELU()
        )
        
        self.q = nn.Conv2d(in_channels= input_dims, out_channels= input_dims, kernel_size= 1, bias= False)
        self.k = nn.Conv2d(in_channels= input_dims, out_channels= input_dims, kernel_size= 1, bias= False)
        self.v = nn.Conv2d(in_channels= input_dims, out_channels= input_dims, kernel_size= 1, bias= False)
        
    def filter(self, x, H, W):
        out = torch.fft.fftn(x, dim= (2,3))
        out = torch.roll(out, (H//2, W//2), dims= (2,3))
        out_low = out* self.mask.to(x.device)
        out = torch.abs(torch.fft.ifftn(out_low, dim= (2,3)))
        
        return out
    
    def forward(self, x, h):
        batchsize, n, dims = x.shape
        
        H = h//16
        W = (n-1)//H
        images = x[:,1:,:].view(batchsize, H, W, -1).permute(0,3,1,2)
        
        out = self.conv(images)
        #print(out.shape)
        out = self.filter(out, H, W)
        
        q = self.q(out).view(batchsize, self.input_dims, -1).permute(0,2,1).contiguous().view(batchsize, n-1, self.nums_head, dims// self.nums_head).permute(0,2,1,3)
        k = self.k(out).view(batchsize, self.input_dims, -1).permute(0,2,1).contiguous().view(batchsize, n-1, self.nums_head, dims// self.nums_head).permute(0,2,1,3)
        v = self.v(out).view(batchsize, self.input_dims, -1).permute(0,2,1).contiguous().view(batchsize, n-1, self.nums_head, dims// self.nums_head).permute(0,2,1,3)
        
        return [q, k, v]

class Audio_Queried_CrossModal_Inconsistency_Enhancer(nn.Module):
    def __init__(self, num_tokens= 4, input_dims= 768, reduction= 4):
        super().__init__()
        
        self.input_dims = input_dims
        self.num_tokens = num_tokens
        
        self.cross_token = nn.Parameter(torch.randn(num_tokens, input_dims))
        
        self.proj_head = nn.Sequential(
            nn.Linear(input_dims, input_dims// reduction),
            nn.GELU(),
            nn.Linear(input_dims// reduction, input_dims)
        )
        self.norm = nn.LayerNorm(input_dims)

    def forward(self, audio, video): #audio: [b, ma, dims]; video: [b*mv, t, dims]
        batchsize, nums_audio_with_head, _= audio.shape
        batch_and_mv, frames, _ = video.shape
        
        nums_video_with_head = batch_and_mv// batchsize
        
        video_nohead = video.view(batchsize, nums_video_with_head, frames, -1)[:, 1:].clone()
        video_nohead = video_nohead.view(-1, frames, self.input_dims) # [b*(mv-1), t, dims]
        
        audio_nohead = audio[:,1:] # [b, ma-1, dims]
        audio_query = repeat(audio_nohead, 't m d -> (t k) m d', k= nums_video_with_head- 1)
        
        rep_token = repeat(self.cross_token, 't d -> b t d', b= (nums_video_with_head- 1)* batchsize)
        cross_attn = torch.bmm(rep_token, audio_query.permute(0,2,1)).softmax(dim= -1)
        rep_token_res = torch.bmm(cross_attn, audio_query)
        
        rep_token = rep_token + rep_token_res
        
        video_attn = torch.bmm(video_nohead, rep_token.permute(0,2,1)).softmax(dim= -1)
        video_res = torch.bmm(video_attn, rep_token)
        
        video_res = video_res.view(batchsize, nums_video_with_head- 1, frames, self.input_dims)
        padded = torch.zeros([batchsize, 1, frames, self.input_dims], device= audio.device)
        
        out = torch.cat([padded, video_res], dim= 1).view(batch_and_mv, frames, -1)
        
        out = torch.add(out, video)
        out = self.norm(out)
        out = self.proj_head(out)
        
        return out

class AudioVisual_Transformer_with_Adapter_Head(nn.Module):
    def __init__(self, num_tokens= 4, input_dims= 768, use_bn= False, reduction= 4, frames= 16):
        super().__init__()
        
        self.input_dims = input_dims
        self.frames = frames
        
        vit_base = create_model('vit_base_patch16_224', pretrained= False, 
                     checkpoint_path= '/home/nief/code/visual-audio/jx_vit_base_p16_224-80ecf9dd.pth', dynamic_img_size= True)
        self.vit_base = copy.deepcopy(vit_base)
        for i in range(len(vit_base.blocks)):
            vit_block = self.vit_base.blocks[i]
            my_block = Block_qkv_Bias(dim= input_dims, num_heads= 12, qkv_bias= True)
            my_block.load_state_dict(vit_block.state_dict(), strict= True)
            self.vit_base.blocks[i] = my_block
            
        hidden_list = []
        for idx, blk in enumerate(self.vit_base.blocks):
            hidden_d_size = blk.mlp.fc1.in_features
            hidden_list.append(hidden_d_size)
        
        self.ViT_Video_Spatio_Attn_Adapter = nn.ModuleList([
            QKV_Adapter_ViT_spatio_filter(input_dims= input_dims, nums_head= 12, reduction= reduction)
            for i in range(len(vit_base.blocks))
        ])
        
        self.Video_ffn_Adapter = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dims, input_dims// reduction),
                nn.GELU(),
                nn.Linear(input_dims// reduction, input_dims)
            )
            for i in range(len(vit_base.blocks))
        ])
        
        self.ViT_Audio_Attn_Adapter = nn.ModuleList([
            QKV_Adapter_ViT_audio_filter(input_dims= input_dims, nums_head= 12, reduction= reduction)
            for i in range(len(vit_base.blocks))
        ])
        
        self.Audio_ffn_adapter = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dims, input_dims// reduction),
                nn.GELU(),
                nn.Linear(input_dims// reduction, input_dims)
            )
            for i in range(len(vit_base.blocks))
        ])
        
        self.query_enhancer = nn.ModuleList([
            Audio_Queried_CrossModal_Inconsistency_Enhancer(num_tokens, input_dims, reduction)
            for i in range(len(vit_base.blocks))
        ])
        
        self.norm_video = nn.LayerNorm(input_dims)
        self.norm_audio = nn.LayerNorm(input_dims)
        
        self.fusion_proj = nn.Linear(input_dims*2, input_dims)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_dims, 2)
        )
        
    def forward_patchEmbed(self, x):
        x = self.vit_base.patch_embed(x)
        x = self.vit_base._pos_embed(x)
        x = self.vit_base.patch_drop(x)
        x = self.vit_base.norm_pre(x)

        return x

    def forward(self, audio, video):
        b, c, h, w = audio.shape
        fa = self.forward_patchEmbed(audio)
        fv = self.forward_patchEmbed(rearrange(video, 'b t c w h -> (b t) c w h'))
        b, ma, dims = fa.shape
        bt, mv, dims = fv.shape
        
        for idx, blk in enumerate(self.vit_base.blocks):
            spatio_attn = self.ViT_Video_Spatio_Attn_Adapter[idx](fv)
            fv_res = blk.drop_path1(blk.ls1(blk.attn(blk.norm1(fv), spatio_attn, True)))
            fv = fv + fv_res
            
            audio_attn = self.ViT_Audio_Attn_Adapter[idx](fa, h)
            fa_res = blk.drop_path1(blk.ls1(blk.attn(blk.norm1(fa), audio_attn, True)))
            fa = fa + fa_res
            
            fv_temporal = fv.view(b, self.frames, mv, dims).permute(0,2,1,3).contiguous().view(-1, self.frames, dims) # [b*mv, t, dims]
            fv_cross_audio = self.query_enhancer[idx](fa, fv_temporal)
            fv_temporal = fv_temporal + fv_cross_audio
            fv = fv_temporal.view(b, mv, self.frames, dims).permute(0,2,1,3).contiguous().view(-1, mv, dims)
            
            fv_norm = blk.norm2(fv)
            fv = fv + blk.drop_path2(blk.ls2(blk.mlp(fv_norm))) + self.Video_ffn_Adapter[idx](fv_norm)
            
            fa_norm = blk.norm2(fa)
            fa = fa + blk.drop_path2(blk.ls2(blk.mlp(fa_norm))) + self.Audio_ffn_adapter[idx](fa_norm) 
        
        fv = self.norm_video(fv) # [b*t, nums, dims]
        fa = self.norm_audio(fa) # [b, nums, dims]
        
        fv_head = fv[:,0].view(b, self.frames, dims).mean(dim= 1)
        fa_head = fa[:,0]
        
        out = torch.cat([fv_head, fa_head], dim= -1)
        fusion_out = self.fusion_proj(out)
        logits = self.fc(fusion_out)
        return logits, fusion_out, fv_head, fa_head

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
if __name__ == '__main__':
    audio = torch.rand([16,3,64,80])
    video = torch.rand([16,12,3,224,224])
    #net = AudioVisual_Transformer_with_Adapter_Head(frames= 12)
    net = AudioVisual_Transformer_with_Adapter_Head(num_tokens= 4, input_dims= 768, use_bn= False, reduction= 4, frames= 16)
    ignored_params = list(map(id, net.vit_base.parameters()))
    adapter_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
    for name, param in net.vit_base.named_parameters():
        param.requires_grad = False
    
    total = []
    for param in adapter_params:
        #if param.requires_grad == True:
        total.append(param.nelement())
    
    print('Number of parameter: % .4fM' % (sum(total) / 1e6))
    