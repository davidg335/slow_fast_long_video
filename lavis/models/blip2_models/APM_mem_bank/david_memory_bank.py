import os 
import numpy as np 
import torch 
from einops import rearrange, reduce, repeat 
import glob 
import glob 
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
# import timm
from transformers import AutoImageProcessor, AutoModel
import torch
import os
import torch
import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import glob 
from pathlib import Path 
import cv2
from einops import rearrange, reduce, repeat
import torch.nn.functional as F
import torch.nn as nn
import math 
from lavis.models.blip2_models.APM_mem_bank.positional_encoding import PositionalEncoding3D 





############### code for tokenizing the frames #########################
"""
from models.tokenizer import Tokenizer
tokenizer = Tokenizer(
    drop_policy='rlt',
    drop_param=0.1,
    encode_length=False,
    num_frames=10,
    tubelet_size=1,
    patch_size=(14, 14),
    frame_size=(448, 448),
    transform=None,
    embed_dims=1024,
    mixup_fn=None,
    random_erase_fn=None,
    rand_aug_fn=None
)
"""
# x : b,t,c,h,w

# b t h w c
"""
def tokenize_frames(frames):
    
    x = rearrange(frames, 'b t c h w -> b t h w c')  
    output = tokenizer(x, None) #no target lot 
    token_mask = output['token_mask']
    print("token_mask shape", token_mask.shape)
    # return dim: b (t h w)
    
    return token_mask
"""
class ApmMemoryBankModel(nn.Module):
    # h,w of the image which will be fwd pass. 
    # coordinate based query 
    def __init__(self, hidden_dim = 1024, t = 32, h = 32, w = 32, fwd_chunk_size = 16):
        
        super(ApmMemoryBankModel, self).__init__()
        
        self.P = h*w #patch size
        self.C = hidden_dim
        self.fc1 = nn.Linear(hidden_dim+h*w, self.P)
        
        self.fc2 = nn.Linear(self.P,self.C, bias = False)
        self.fc3 = nn.Linear(self.C,self.P, bias = False)
        self.fc4 = nn.Linear(self.P, self.C, bias = False)
        self.fc5 = nn.Linear(self.C,self.P, bias = False)
        self.fc6 = nn.Linear(self.P, self.C, bias = False)
        # self.feat_proj_head = nn.Linear(1024, 1024)
        
        self.rgb_head_1 = nn.Linear(1024*3, 256)
        self.rgb_head_2 = nn.Linear(256,256)
        self.rgb_head_3 = nn.Linear(256,3)

        #initialize positional encoding 
        # self.pos = positionalencoding2d(hidden_dim, h,w) #to break input coordinate symmetry
        self.hidden_dim = hidden_dim
        self.h, self.w = h,w
        self.t = t
        self.fwd_chunk_size = fwd_chunk_size
        
        #init a single patch size 
        #will operate on 448 by 448 to get information into the columns
        self.patch_size = 14
        self.stride = 14
        self.conv1 = nn.Conv2d(3, 1, kernel_size=self.patch_size, stride=self.stride)
        
        
    def init_positional_encoding(self, start_time ):
        pos_3d = PositionalEncoding3D(self.hidden_dim).cuda()
        t, h, w = self.t, self.h, self.w        
        fw = torch.zeros(1,t,h,w,self.hidden_dim).cuda()
        print("called pos_3d....")
        pos = pos_3d.forward(fw, start_time = start_time).squeeze(0)
        pos = rearrange(pos, 't h w c -> (t h w) c')
        pos = pos.cuda() # h*w*l , 1024 
        #now we have to return the pos 
        return pos 

    def forward_chunk(self, x):
        x_pos = x#contains the whole cortical column stack
        
        x = self.fc1(x)
        x = F.relu(x)
        #add layer norm 
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        
        feat = self.fc6(x)
        
        # rgb = F.relu(self.rgb_head_1(torch.cat([feat,x_pos],1))) #breaks rgb output symmetry
        # rgb = F.relu(self.rgb_head_2(rgb))
        # rgb = F.relu(self.rgb_head_3(rgb))
        
           
        # return feat, rgb # feat is the feature which is produced
        return feat, None

    # x is frames
    # x : b,t,c,h,w
    # feat = b t (hw) d
    # pos = (t h w) d 
    #token_mask b (t h w)
    # return the weights in which the information is compressed, these form the queries that the transformer will self attend
    def get_model_unfolded_params(self):
        # p1 is skipped thats just 
        p2 = self.fc2.weight.data.T    # [C, P] → [P, C]
        p3 = self.fc3.weight.data      # already [P, C]
        p4 = self.fc4.weight.data.T    # [C, P] → [P, C]
        p5 = self.fc5.weight.data      # already [P, C]
        p6 = self.fc6.weight.data.T    # [C, P] → [P, C]
        
        print(f"Model unfolded params:{torch.stack([p2, p3, p4, p5, p6]).shape}")
        return torch.stack([p2, p3, p4, p5, p6])  # shape: [5, P, C]
        
    
    # x is all of the frames of the vid
    def forward_wrapper(self,x, feat, pos, token_mask=None ):
        # print("krishna", x.shape)
        # print("!", feat.shape)
        # print("pos", pos.shape)
        b, t, c, h, w = x.shape #h,w are 448. This is the original image
        #rearange and run conv 2d 
        #print("x shape at line 160:",x.shape) #torch.Size([1, 1, 3, 224, 224])
        x = rearrange(x, 'b t c h w -> (b t) c h w')  # shape: [32, 3, 448, 448]
        x = self.conv1(x) # shape: [32, 3, 32, 32]
        # print("x shape", x.shape)
        #print("x shape at line 164:",x.shape) #torch.Size([1, 1, 16, 16])
        x = rearrange(x, '(b t) c h w -> (b t) (h w) c', t = t, h = self.h, w= self.w) #shape: [32,1024,3]
        x = x.squeeze(-1) # squeeze channel 
        # print("x shape", x.shape)
        feat=feat[:, :, :-1, :]  # removes the last token, the cls token from Vision encoder output

        
        summary_feat = repeat(x, '(b t) d -> (b t)  h w d', t = t, h = self.h, w = self.w)
        summary_feat = rearrange(summary_feat, '(b t) h w d ->  b t h w d', b = b, t = t, h = self.h, w = self.w)

        # print("summary_feat shape", summary_feat.shape)
        pos= rearrange(pos, '(t h w) d ->  t h w d', t = self.t, h = self.h, w = self.w)
        pos=pos[0:t,:,:,:]
        pos = repeat(pos, 't h w d -> b t h w d', b = b)
        
        #print("pos shape", pos.shape)
        #print("summary_feat shape", summary_feat.shape) #summary_feat shape torch.Size([1, 1, 16, 16, 256]), want the 256 to be 768


        input_feat = torch.cat([summary_feat, pos], dim=-1)
        input_feat = rearrange(input_feat, 'b t h w d -> (b t h w) d')  # shape: [1, 32, 32, 32, 1024]        
        #token_mask = rearrange(token_mask, 'b (t h w) -> (b t h w)', t = t, h = self.h, w = self.w).cuda()
        target_feat = rearrange(feat, 'b t (hw) d -> (b t hw) d')
        #print("input feat shape", input_feat.shape) #[4096, 1664])

        # token_mask = token_mask.bool()  # in case it's 0/1
        # input_feat = input_feat[token_mask]  # filter out the masked tokens
        # target_feat = target_feat[token_mask]

        token_mask = torch.zeros((h // self.patch_size) * (w // self.patch_size) * t, dtype=torch.bool)
        #token_mask = torch.zeros_like(token_mask)  # reset token_mask to all zeros, as we don't need it anymore
        #set random 2000 locations to 1
        num_ones = 2000
        indices = torch.randperm(token_mask.numel())[:num_ones]
        token_mask[indices] = 1
        # print(np.sum(token_mask.cpu().numpy()), "tokens selected",indices)
        input_feat = input_feat[token_mask]  # filter out the masked tokens
        target_feat = target_feat[token_mask]
        # # Set selected indices to 1
        #print("input_feat shape after token mask", input_feat.shape) #[, 1664])

        chunk_size = self.fwd_chunk_size
        n_chunks = input_feat.shape[0] // chunk_size
        if input_feat.shape[0] % chunk_size != 0:
            n_chunks += 1
        n_forwards = 0
        for i in range(n_chunks):
            #print(f"Chunk {i}/{n_chunks}, start {i*chunk_size}")
            
            start = i*chunk_size
            end = min((i+1)*chunk_size, input_feat.shape[0])
            input_feat_chunk = input_feat[start:end]
            # print("before forward", input_feat_chunk.shape)
            
            feat_chunk, _ = self.forward_chunk(input_feat_chunk)
            # print("forward done", feat_chunk.shape)
            # exit(1) 
            n_forwards+=1
            if i == 0:
                feat_out = feat_chunk
                rgb_out = None
            else:
                feat_out = torch.cat([feat_out, feat_chunk], dim=0)
                rgb_out = None #torch.cat([rgb_out, rgb_chunk], dim=0)
        #print("shape of feat_out",feat_out.shape)
        #print("shape of target_feat",target_feat.shape)
        

        feat_loss = F.mse_loss(feat_out, target_feat)
        # print("outtttt", feat_out.shape, target_feat.shape)
        # exit(1)
        # rgb_loss = F.mse_loss(rgb_out, target_rgb)
        loss = feat_loss #+ rgb_loss
        
        # rgb_out = rearrange(rgb_out, '(b h w) c -> b c h w', b = b, h = self.h, w = self.w)
        
        return loss    #, feat_loss, rgb_loss, feat_out, rgb_out

        # print("token_mask shape", token_mask.shape, torch.sum(token_mask))
        # print("input_feat shape", input_feat.shape)
        # print("target_feat shape", target_feat.shape)    

if __name__ == "__main__":

    B = 1
    T, H, W = 10, 448, 448
    patch_size = 14

    model = ApmMemoryBankModel(hidden_dim=1024, t=T, h=H//patch_size, w=W//patch_size, fwd_chunk_size=16)
    model = model.cuda()
    
    model.load_state_dict(torch.load('memory_bank_model.pth'), strict=True)
    print("loaded weights")
    # exit(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    """
    # read 16 frames 
    root = Path('./')/'video_resized'
    paths = glob.glob(str(root/'**/*.jpg'), recursive = True)
    
    feat_dir = Path('./')/'features'
    
    all_paths = sorted(paths)
    """
    # n_slices = len(all_paths)// T
    # print("n_slices", n_slices)
    n_slices = 10
    steps_per_slice = 300
    
    for i in range(n_slices):
        print("start indx", i*T, "end index", (i+1)*T)
        paths = all_paths[i*T:(i+1)*T]
        t = i * T  # start time for time-shifted pos-encoding
        
        # ##### for the first T time slices
        # t = 0 # start time for time-shifted pos-encoding # will generate a T,H,W,C tensor
        
        # paths = paths[:T]
        frames = []
        feats = []
        
        
        for path in paths:
            image = Image.open(path).convert("RGB")
            image_id = str(Path(path)).split('/')[-1].split('.')[0]
        
            #Resize using BILINEAR interpolation
            # image = image.resize((448, 448), resample=Image.BILINEAR)
            frames.append(image)
            feature_path = feat_dir / f"{image_id}.npy"
            feat = np.load(feature_path)
            feat = torch.from_numpy(feat).float()
            feats.append(feat)
        
        feat = torch.stack(feats, dim=0) 
        # feat = feat.unsqueeze(0).cuda()  # add one batch dimension
        frames = torch.stack([transforms.ToTensor()(frame) for frame in frames], dim=0)  # shape: [n_frames, 3, H, W]
        frames = frames.unsqueeze(0)  # shape: [1, n_frames, 3, H,W]
        #normalize the frames to Imagenet Stats
        imagenet_mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=frames.dtype, device=frames.device)
        imagenet_std = torch.as_tensor([0.229, 0.224, 0.225], dtype=frames.dtype, device=frames.device)
        frames = (frames - imagenet_mean[None, :, None, None]) / imagenet_std[None, :, None, None]
        frames = frames.cuda()
        
        # print("framess shape before tokenizer", frames.shape)
        # exit(1)
        # frames = rearrange(frames, 'b t c h w ->(b t)c  h w')
        
        
        pos = model.init_positional_encoding(start_time = t)
        #token_mask = tokenize_frames(frames)
        # print("feat shape", feat.shape)
        # exit(1)
        feat = feat.unsqueeze(0).cuda()  # add one batch dimension
        memory_bank_weights = model.get_model_unfolded_params()
        
        
        avg_loss = []
        
        # # 
        #steps_per_slice is the number of training epochs we do for each image 
        #frames is the set of images in the batch we are processing
        print(f"Shape of frames: {frames.shape}")
        print(f"Shape of feat: {frames.shape}")

        for j in range(steps_per_slice):    
            optimizer.zero_grad()
            loss = model.forward_wrapper(frames, feat, pos, token_mask)
            loss.backward()
            optimizer.step()
            avg_loss.append(loss.item())
            #take last 10 entries and avg them 
            print(f"Slice {i+1}/{n_slices}, Step {j+1}/{steps_per_slice}, Loss: {np.average(avg_loss[-10:])}")
        
        
        torch.save(model.state_dict(), 'memory_bank_model.pth')
        
        #### just eval code ########################################
        loss = model.forward_wrapper(frames, feat, pos, token_mask)
        l=model.get_model_unfolded_params()
        [p2,p3,p4,p5,p6] = l
        print(p2.shape)
        print(p3.shape)
        print(p4.shape)
        print(p5.shape)
        print(p6.shape)
        
        print("Slice", i+1, "/", n_slices, "Loss:", loss.item())    
        exit(1)
        ############################################################
    # torch.save(model.state_dict(), 'memory_bank_model.pth')

        # print(frames.shape)
        #how to run 2d conv of patch size 14 on the frames
        # print(feat.shape)
        
        
        ################# just checking that the time shifting in pos enc is correct ###############   
        # ###### initialize the positional encoding whose start time changes with the offset
        # pos_1 = model.init_positional_encoding(start_time = 0)
        # # print("pos shape", pos_1.shape)
        # pos_2 = model.init_positional_encoding(start_time = 3)
        # # print("pos 2", pos_2.shape)
        
        # # pos_1 = rearrange(pos_1, ' (b t h w) c -> b t h w c', t = T, h = H//model.patch_size, w = W//model.patch_size)
        # # pos_2 = rearrange(pos_2, ' (b t h w) c -> b t h w c', t = T, h = H//model.patch_size, w = W//model.patch_size)
        # # print("-----1", pos_1[0][3][0][0])
        # # print("-----2", pos_2[0][0][0][0])

        
        # pass  