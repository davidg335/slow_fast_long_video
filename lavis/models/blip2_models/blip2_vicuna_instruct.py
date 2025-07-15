"""
Requires Transformer 4.28 and above, implementation may change according the Llama implementation
"""
import logging
import string
from packaging import version

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

import transformers

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.Qformer import memory_bank_compress

from lavis.models.blip2_models.APM_mem_bank.david_memory_bank import ApmMemoryBankModel
import numpy as np

@registry.register_model("blip2_vicuna_instruct_malmm")
class Blip2VicunaInstruct_MALMM(Blip2Base):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "vicuna7b": "configs/models/blip2/blip2_instruct_vicuna7b.yaml",
        "vicuna13b": "configs/models/blip2/blip2_instruct_vicuna13b.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        llm_model="",
        prompt="",
        max_txt_len=8,
        max_output_txt_len=256,
        apply_lemmatizer=False,
        qformer_text_input=True,
        memory_bank_length=0,
        num_frames=0,
        max_num_frames=120,
    ): # memory_bank_length is the max size of the memory bank. 
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.28"), "BLIP-2 Vicuna requires transformers>=4.28"        
        from transformers import LlamaTokenizer
        from lavis.models.blip2_models.modeling_llama import LlamaForCausalLM
        
        self.tokenizer = self.init_tokenizer(truncation_side="left")
        self.img_size=img_size
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        """
        #trying to figure out hidden dimension of the visual encoder
        device = next(self.visual_encoder.parameters()).device
        dtype = next(self.visual_encoder.parameters()).dtype  # Ensures correct fp16 or fp32
        
        dummy_input = torch.randn(1, 3, img_size, img_size, device=device, dtype=dtype)
        with torch.no_grad():
            output = self.visual_encoder(dummy_input)
        print(f"Vision encoder output shape: {output.shape}")
        print(f"hidden dimension:{output.shape[-1]}")
        self.hidden_dim=output.shape[-1]
        """
        self.hidden_dim=1408
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        print(f"Inputs to the Qformer... \n num_query_token {num_query_token} \n self.visual_encoder.num_features {self.visual_encoder.num_features} \n memory_bank_length {memory_bank_length} \n num_frames {num_frames}")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, memory_bank_length=memory_bank_length, num_frames=num_frames,
        )
        print(f"qformer_text_input value:{qformer_text_input}")
        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None

        self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side="left")
        
        self.llm_model = LlamaForCausalLM.from_pretrained(
            llm_model, torch_dtype=torch.float16
        )
        """  
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"  # Optional; depends on memory setup
        )
        """

        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        # self.llm_tokenizer.pad_token = self.llm_tokenizer.unk_token

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        # self.eos_token_id = self.llm_tokenizer(
        #     self.llm_tokenizer.eos_token, add_special_tokens=False
        # ).input_ids[0]

        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt
        prompt_tokens = self.llm_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        self._lemmatizer = None

        self.qformer_text_input = qformer_text_input
        self.num_query_token = num_query_token
        self.memory_bank_length = memory_bank_length
        self.use_memory_bank = True if memory_bank_length > 0 else False
        self.num_frames = num_frames
        self.visual_memory_bank = None
        self.image_pe = nn.Embedding(max_num_frames, 1408)
        nn.init.constant_(self.image_pe.weight, 0.0)
        print("init for Blip2")
        #initialize the APM memory bank
        print("about to instantiate apm model ...")
        self.forward_chunk_size=16
        self.apm_mem_bank_model = ApmMemoryBankModel(hidden_dim=self.hidden_dim, t = 16, h = self.img_size//14, w = self.img_size//14, fwd_chunk_size = self.forward_chunk_size) 
        self.apm_mem_bank_model = self.apm_mem_bank_model.cuda() 
        self.apm_optimizer = torch.optim.Adam(self.apm_mem_bank_model.parameters(), lr=1e-3)
        print("apm made...")



    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len

    def forward(self, samples):
        # print('-----------------')
        # print(samples["text_input"])
        # print(samples["text_output"])
        # print('-----------------')
        print("entering forward pass")
        image = samples["image"]
        # For video data
        is_video = False
        if image.dim() == 5:
            is_video = True
            B, C, T, H, W = image.shape #batch, channel (rgb), timestep, frame height, frame width
            print(f"Image Shape: {image.shape}") # [8,3,20,224,224]
            print(f"Image info:{image[:,0,-1,0,0]}")
        if self.qformer_text_input:
            if is_video:
                text_input = [text for text in samples["text_input"] for _ in range(T)]
            else:
                text_input = samples["text_input"]

            if self.use_memory_bank:
                query_tokens = self.query_tokens.expand(B, -1, -1) # [B, 32, C]
                text_Qformer = self.tokenizer(
                    samples["text_input"], # [B]
                    padding='longest',
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)
                query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device) # [B, N], N=32
                Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)
                apm_atts=torch.ones(B,5*(self.img_size//14)*(self.img_size//14), dtype=torch.long).to(image.device) # [B,256*5]
                print("Apm_atts shape",apm_atts.shape)
                ########APM#######################
                print("Starting APM training...")
                #get the image embeddings for the whole video
                with torch.no_grad():
                    with self.maybe_autocast():
                        sample_embeds = self.ln_vision(self.visual_encoder(image[:, :, 0, :, :]))  # [B, 257, 1408]
                        N, C = sample_embeds.shape[-2:]  # N = 257, C = 1408

                image_embeds_wholevid = torch.zeros(B, T, N, C, device=image.device, dtype=sample_embeds.dtype)  # [B, T, 257, 1408]
                image_atts_wholevid = torch.zeros(B, T, N, device=image.device, dtype=torch.long)  # [B, T, 257]
                # for video vid, store embeddings for each frame in image_embeds_wholevid
                for t in range(T):
                    with self.maybe_autocast():
                        image_embeds_t = self.ln_vision(self.visual_encoder(image[:, :, t, :, :]))  # [B, 257, 1408]
                        
                    position_ids = torch.tensor([t]).long().to(image_embeds_t.device) #[1]
                    position_ids = position_ids.unsqueeze(0).expand(B, -1) #[B, 1]
                    image_embeds_t = image_embeds_t + self.image_pe(position_ids) # [B, N, C]
                    image_atts = torch.ones(image_embeds_t.size()[:-1], dtype=torch.long).to(image.device) # [B, N], N is 257
                    #image_embeds_t = image_embeds_t.unsqueeze(1) # [B, 1, N, C]

                    image_embeds_wholevid[:, t, :, :] = image_embeds_t
                    image_atts_wholevid[:,t,:]=image_atts
                
                APM_weights = torch.zeros(B, 5, N-1, C, device=image.device, dtype=sample_embeds.dtype)  # Shape [B,5,N-1,C], N=257
                imagenet_mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=image.dtype, device=image.device)
                imagenet_std = torch.as_tensor([0.229, 0.224, 0.225], dtype=image.dtype, device=image.device)
                n_slices = T//self.forward_chunk_size+1 # set this equal to T//16
                for vid in range(B):
                    print("video",vid)
                    #steps_per_slice = 10 #same as num epochs
                    for i in range(n_slices):
                        #############################################
                        #############################################
                        ################APM-CODE#####################
                        t=i*self.forward_chunk_size
                        print(f"We are at timestep t={t}")
                        if t< self.memory_bank_length:
                            #do 5 iterations of training
                            steps_per_slice=2
                        else:
                            #do 2 iterations of training
                            steps_per_slice=2
                        # image shape B, C, T, H, W, image[:, :, t, :, :] 
                        #frames=image[:, :, t, :, :] # want shape [B,1,C,H,W],torch.Size([1, 10, 3, 448, 448])
                        frames = image[vid:vid+1, :, i*self.forward_chunk_size:(i+1)*self.forward_chunk_size, :, :]  #BCTHW     # shape: [1,C,self.forward_chunk_size, H, W] 
                        frames = frames.permute(0, 2, 1, 3, 4)  # shape: [1, self.forward_chunk_size, C, H, W]
                        frames = (frames - imagenet_mean[None, :, None, None]) / imagenet_std[None, :, None, None]
                        frames = frames.cuda()
                        #print(f"Frames shape: {frames.shape}")
                        feat=image_embeds_wholevid[vid:vid+1,i*self.forward_chunk_size:(i+1)*self.forward_chunk_size,:,:] #[B, T, 257, 1408], want  ([1, self.forward_chunk_size, 256, 1408]), good
                        pos = self.apm_mem_bank_model.init_positional_encoding(start_time = t)
                        token_mask=None
                        #print(f"Shape of feat:{feat.shape}")
                        #print(f"Shape of pos:{pos.shape}")
                        #print(f"Shape of frames:{frames.shape}")
                        
                        #update APM weights based on new frame
                        avg_loss = []
                        with torch.set_grad_enabled(True):
                            for j in range(steps_per_slice):
                                self.apm_optimizer.zero_grad()
                                apm_loss = self.apm_mem_bank_model.forward_wrapper(frames.detach(), feat.detach(), pos.detach(), token_mask)
                                apm_loss.backward()
                                for name, param in self.apm_mem_bank_model.named_parameters():
                                    print(param)
                                    if param.grad is not None and torch.isnan(param.grad).any():
                                        print(f"NaN gradient in {name}")
                                self.apm_optimizer.step()
                                avg_loss.append(apm_loss.item())
                                #take last 10 entries and avg them 
                                print(f"Slice {t+1}/{T}, Step {j+1}/{steps_per_slice}, Loss: {np.average(avg_loss[:])}")
                    #save weights at the end
                    with torch.no_grad():
                        apm_memory_bank_weights = self.apm_mem_bank_model.get_model_unfolded_params().detach() # shape is torch.Size([5, 256, 1408])
                    APM_weights[vid,:,:,:]=apm_memory_bank_weights
                #now, freeze the APM
                ###############################
                ###############################
                ###############################
                ###############################
                #torch.save(self.apm_mem_bank_model.state_dict(), 'apm_memory_bank_model.pth')

                #forward pass through visual mem bank, qformers for each frame...
                #here, T is the number of frames
                
                for t in range(T):
                    #print("frame",t)
                    image_embeds=image_embeds_wholevid[:,t:t+1,:,:] # [B,1,N,C]
                    image_atts=image_atts_wholevid[:,t,:] # [B,N]
                    if t == 0:
                        encoder_hidden_states = image_embeds # [B, 1, N, C]
                        self.size_constant = torch.ones(B, 1, N).to(image_embeds.device) # [B, 1, N]
                    else:
                        encoder_hidden_states = torch.cat([self.visual_memory_bank, image_embeds], dim=1) # [B, (t+1), N, C], this is the input the cross attention step, will act as both Key and Value
                    
                    query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=encoder_hidden_states.view(B, -1, C),
                        apm_hidden_states=APM_weights.view(B, -1, C),
                        apm_attention_mask=apm_atts,
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                    )

                    # If it is the first frame, initialize the visual_memory_bank as the embedding of the first frame
                    # If not, concatenate the visual_memory_bank with the current frame embedding and update the compression_size
                    if t == 0:
                        self.visual_memory_bank = image_embeds.detach()  # [B, 1, N, C]
                        self.compression_size = self.size_constant  # [B, 1, N]
                    else:
                        self.visual_memory_bank = torch.cat([self.visual_memory_bank, image_embeds.detach()], dim=1)  # [B, t+1, N, C]
                        self.compression_size = torch.cat([self.compression_size, self.size_constant], dim=1)  # [B, t+1, N]
                    
                    # If it is the last frame, delete the visual_memory_bank and compression_size
                    # Else, if the current length of the visual_memory_bank exceeds the threshold, compress the visual_memory_bank
                    if t == T - 1:
                        del self.visual_memory_bank
                        del self.compression_size
                    elif self.visual_memory_bank.size(1) > self.memory_bank_length:
                        self.visual_memory_bank, self.compression_size = memory_bank_compress(self.visual_memory_bank, self.compression_size)
                    

            
                    
            #this is if there is no mem bank... ignore...
            else:
                query_tokens = self.query_tokens.expand(B * T, -1, -1)
                text_Qformer = self.tokenizer(
                    text_input,
                    padding='longest',
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)
                query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
                Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

                if is_video:
                    image = image.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                with self.maybe_autocast():
                    image_embeds = self.ln_vision(self.visual_encoder(image)) #[B * T, 256+1, 1408]
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
        else:
            if is_video:
                image = image.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image)) #[B * T, 256+1, 1408]
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            query_tokens = self.query_tokens.expand(B * T, -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        if is_video:
            inputs_llm = inputs_llm.reshape(B, -1, inputs_llm.shape[-1])
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            samples['text_input'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        self.llm_tokenizer.truncation_side = 'right'
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in samples['text_output']],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        ).to(image.device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        # do not apply loss to the padding
        targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
        )

        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        # do not apply loss to the query tokens
        empty_targets = (
            torch.ones(atts_llm.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
        inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)

        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        loss = outputs.loss
        
        #########################################################
        #save the weights for the apm model
        #torch.save(self.apm_mem_bank_model.state_dict(), 'memory_bank_model.pth')
        #########################################################
        
        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        print("generating!!")
        self.llm_tokenizer.padding_side = "left"

        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        image = samples["image"]
        # For video data
        is_video = False
        if image.dim() == 5:
            is_video = True
            B, C, T, H, W = image.shape
            print(f"Image Shape: {image.shape}") # [8,3,20,224,224]
            print(f"Image info:{image[:,0,-1,0,0]}")
            #print("The shape of the frames/images for vid",image.shape)

        if isinstance(prompt, str):
            prompt = [prompt] * B
        else:
            assert len(prompt) == B, "The number of prompts must be equal to the batch size."

        # For TextCaps
        if "ocr_tokens" in samples.keys() and "{}" in prompt[0]:
            prompt = [p.format(', '.join(samples['ocr_tokens'][i][:30])) for i, p in enumerate(prompt)]

        assert self.qformer_text_input == True
        if self.qformer_text_input:
            # remove ocr tokens in q_former (for eval textvqa)
            # qformer_prompt = prompt
            # qformer_prompt = ['Question: ' + qp.split(' Question: ')[1] for qp in qformer_prompt]

            if is_video:
                text_input = []
                for text in prompt:
                    text_input.extend([text] * T)
            else:
                text_input = prompt

            if self.use_memory_bank:
                query_tokens = self.query_tokens.expand(B, -1, -1)  # [B, 32, C]
                text_Qformer = self.tokenizer(
                    prompt,  # [B]
                    padding='longest',
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)
                query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)  # [B, 32]
                Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)
                for t in range(T):
                    with self.maybe_autocast():
                        image_embeds = self.ln_vision(self.visual_encoder(image[:, :, t, :, :])).detach()  # [B, 256+1, 1408]
                    N, C = image_embeds.shape[-2:]
                    position_ids = torch.tensor([t]).long().to(image_embeds.device)  # [1]
                    position_ids = position_ids.unsqueeze(0).expand(B, -1)  # [B, 1]
                    image_pe = self.image_pe(position_ids)  # [B, 1, C]
                    image_embeds = image_embeds + image_pe  # [B, N, C]
                    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)  # [B, N]
                    image_embeds = image_embeds.unsqueeze(1)  # [B, 1, N, C]

                    if t == 0:
                        self.visual_memory_bank = image_embeds  # [B, 1, N, C]
                        self.size_constant = torch.ones(B, 1, N).to(image_embeds.device)  # [B, 1, N]
                        self.compression_size = self.size_constant  # [B, 1, N]
                    else:
                        self.visual_memory_bank = torch.cat([self.visual_memory_bank, image_embeds], dim=1)  # [B, t+1, N, C]
                        self.compression_size = torch.cat([self.compression_size, self.size_constant], dim=1)  # [B, t+1, N]
                    #############################################
                    #############################################
                    ################APM-CODE#####################
                    
                    print(f"We are at timestep t={t}")
                    if t< self.memory_bank_length:
                        #do 5 iterations of training
                        steps_per_slice=5
                    else:
                        #do 2 iterations of training
                        steps_per_slice=2
                    
                    # image shape B, C, T, H, W, image[:, :, t, :, :] 
                    #frames=image[:, :, t, :, :] # want shape [B,1,C,H,W],torch.Size([1, 10, 3, 448, 448])
                    frames = image[:, :, t, :, :]           # shape: [B, C, H, W] → 4D
                    frames = frames.unsqueeze(2)            # shape: [B, C, 1, H, W] → 5D
                    frames = frames.permute(0, 2, 1, 3, 4)  # shape: [B, 1, C, H, W]
                    #print(f"Frames shape: {frames.shape}")
                    #frames = frames.permute(0, 2, 1, 3, 4)  # now shape [B, 1, C, H, W]. H,W are original pixel dims of image
                    feat=image_embeds #feature rep for the current frame that we are on, shape is [B, 1, N, C] want [B,1,N-1,C] ([1, 10, 256, 1408]), good
                    pos = self.apm_mem_bank_model.init_positional_encoding(start_time = t)
                    token_mask=None
                    #print(f"Shape of feat:{feat.shape}")
                    #print(f"Shape of pos:{pos.shape}")
                    #print(f"Shape of frames:{frames.shape}")
                    
                    #update APM weights based on new frame
                    avg_loss = []
                    with torch.set_grad_enabled(True):
                        for j in range(steps_per_slice):
                            self.apm_optimizer.zero_grad()
                            loss = self.apm_mem_bank_model.forward_wrapper(frames, feat, pos, token_mask)
                            loss.backward()
                            self.apm_optimizer.step()
                            avg_loss.append(loss.item())
                            #take last 10 entries and avg them 
                            print(f"Slice {t+1}/{T}, Step {j+1}/{steps_per_slice}, Loss: {np.average(avg_loss[-10:])}")
                    #add APM here!
                    apm_memory_bank_weights = self.apm_mem_bank_model.get_model_unfolded_params() # shape is torch.Size([5, 256, 1408]), want shape [B,5, N, C]
                    apm_memory_bank_weights = apm_memory_bank_weights.unsqueeze(0)  # Shape: [1, 5, N, 1024]
                     # want shape [B,5, N, C]
                    repeat_factor=self.memory_bank_length//5
                    #print(f"Memory bank length:{self.memory_bank_length}")
                    apm_memory_bank_weights = apm_memory_bank_weights.repeat(1, repeat_factor, 1, 1) # shape: [B, 20, N, C]
                    ## add padding to vis mem bank so I can add it to the apm
                    _, k, _, _ = self.visual_memory_bank.shape
                    desired_k = self.memory_bank_length #should be 20
                    #print(f"Shape of visual memory bank: {self.visual_memory_bank.shape}")
                    if k < desired_k:
                        pad_size = desired_k - k
                        padding = torch.zeros(B, pad_size, N, C, device=self.visual_memory_bank.device, dtype=self.visual_memory_bank.dtype)
                        visual_memory_bank_padded = torch.cat([self.visual_memory_bank, padding], dim=1)
                    else:
                        visual_memory_bank_padded = self.visual_memory_bank
                    zeros = torch.zeros(B, self.memory_bank_length, 1, C, device=apm_memory_bank_weights.device, dtype=apm_memory_bank_weights.dtype) #[B, 20, 256, C] to [B, 20, 257, C] to match visual mem bank size
                    #print(f"Shape of apm memory bank weights: {apm_memory_bank_weights.shape}")
                    #print(f"Shape of zeros: {zeros.shape}")

                    apm_memory_bank_weights_padded = torch.cat([apm_memory_bank_weights, zeros], dim=-2)
                    ##print(f"Shape of padded visual memory bank: {visual_memory_bank_padded.shape}")
                    #print(f"Shape of padded apm memory bank weights: {apm_memory_bank_weights_padded.shape}")

                    #memory_bank_hidden_states_summed = visual_memory_bank_padded + apm_memory_bank_weights_padded # want shape to be [B, 20, N, C], N=257
                    #print("visual mem bank size: ", self.visual_memory_bank.shape)
                    #############################################
                    #############################################
                    query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=visual_memory_bank_padded.view(B, -1, C),
                        apm_hidden_states=apm_memory_bank_weights.view(B, -1, C),
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                    )
                    last_hidden_state = query_output.last_hidden_state

                    if t == T - 1:
                        del self.visual_memory_bank
                        del self.compression_size
                    elif self.visual_memory_bank.size(1) >= self.memory_bank_length: #changed this from > to >= to accomate APM
                        self.visual_memory_bank, self.compression_size = memory_bank_compress(self.visual_memory_bank,
                                                                                               self.compression_size)

            else:
                query_tokens = self.query_tokens.expand(B * T, -1, -1)
                text_Qformer = self.tokenizer(
                    text_input, # [B*T]
                    padding='longest',
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)
                query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
                Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

                if is_video:
                    image = image.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                with self.maybe_autocast():
                    image_embeds = self.ln_vision(self.visual_encoder(image)) #[B * T, 256+1, 1408]
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
        else:
            if is_video:
                image = image.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image)) #[B * T, 256+1, 1408]
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            query_tokens = self.query_tokens.expand(B * T, -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        if is_video:
            inputs_llm = inputs_llm.reshape(B, -1, inputs_llm.shape[-1])
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        llm_tokens = self.llm_tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt"
        ).to(image.device)

        with self.maybe_autocast():
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

            #print("inputs_embeds:", inputs_embeds.shape)
            #print("attention_mask:", attention_mask.shape, "values in", attention_mask.min().item(), attention_mask.max().item())

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                # eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

        outputs[outputs < 2] = 2 # convert output id -1/0/1 to 2 (eos_token_id)
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return output_text

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=0,
        **kwargs
    ):
        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        if prompt:
            if prompt.count("{}") == 2:
                if 'ocr_tokens' in samples:
                    text_input = [
                        prompt.format(', '.join(samples['ocr_tokens'][i][:30]), samples["text_input"][i])
                    for i in range(len(samples["text_input"]))]
                elif 'choices' in samples:
                    text_input = []
                    for i in range(len(samples["text_input"])):
                        this_choices = [f"({string.ascii_lowercase[j]}) {ch}" for j, ch in enumerate(samples["choices"][i])]
                        this_choices = " ".join(this_choices)
                        text_input.append(prompt.format(samples["text_input"][i], this_choices))
            else:
                text_input = [prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]

        samples["prompt"] = text_input

        output_text = self.generate(
            samples,
            num_beams=num_beams,
            max_length=max_len,
            min_length=min_len,
            length_penalty=length_penalty,
            num_captions=num_beams,
        )

        if "apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]:
            output_text = self._lemmatize(output_text)

        return output_text

    def predict_class(
        self,
        samples,
        candidates,
        n_segments=1,
    ):
        self.llm_tokenizer.padding_side = "left"

        # If candidates is a list of lists, each sample has its candidates, then we need to iterate one by one
        if type(candidates[0]) == list:
            results = []

            for i in range(samples["image"].size(0)):
                this_sample = {
                    "image": samples["image"][i].unsqueeze(0),
                    "prompt": samples["prompt"],
                }

                if "text_input" in samples.keys():
                    this_sample["text_input"] = [samples["text_input"][i]]

                if 'context' in samples.keys():
                    this_sample['context'] = [samples["context"][i]]

                if 'history' in samples.keys():
                    this_sample['history'] = [samples["history"][i]]

                if 'caption' in samples.keys():
                    this_sample['caption'] = [samples["caption"][i]]

                this_result = self._predict_class(this_sample, candidates[i], n_segments)
                results.append(this_result)

            try:
                results = torch.cat(results, dim=0)
            except:
                results = [res.tolist()[0] for res in results]

            return results

        return self._predict_class(samples, candidates, n_segments)

    def _predict_class(
        self,
        samples,
        candidates,
        n_segments=1,
    ):
        image = samples["image"]
        prompt = samples["prompt"]

        bs = image.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        if "text_input" in samples.keys():
            if type(samples["text_input"][0]) == list:
                prompt = [prompt[i].format(*samples["text_input"][i]) for i in range(len(prompt))]
            else:
                prompt = [prompt[i].format(samples["text_input"][i]) for i in range(len(prompt))]

        # scienceqa
        if 'context' in samples.keys() and samples['context'] != '':
            prompt = [f'context: {samples["context"][i]}. {prompt[i]}' for i in range(len(prompt))]

        # visual dialog
        if 'history' in samples.keys() and samples['history'][0] != '':
            prompt = [f'dialog history: {samples["history"][i]}\n{prompt[i]}' for i in range(len(prompt))]

        if 'caption' in samples.keys() and samples['caption'][0] != '':
            prompt = [f'This image has the caption "{samples["caption"][i]}". {prompt[i]}' for i in range(len(prompt))]

        query_tokens = self.query_tokens.expand(bs, -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt"
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        if image.dim() == 5:
            inputs_llm, atts_llm = [], []
            for j in range(image.size(2)):
                this_frame = image[:,:,j,:,:]
                with self.maybe_autocast():
                    frame_embeds = self.ln_vision(self.visual_encoder(this_frame))
                    frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(image.device)

                if self.qformer_text_input:
                    frame_query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                else:
                    frame_query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )

                frame_inputs_llm = self.llm_proj(frame_query_output.last_hidden_state[:,:query_tokens.size(1),:])
                frame_atts_llm = torch.ones(frame_inputs_llm.size()[:-1], dtype=torch.long).to(image.device)
                inputs_llm.append(frame_inputs_llm)
                atts_llm.append(frame_atts_llm)
            inputs_llm = torch.cat(inputs_llm, dim=1)
            atts_llm = torch.cat(atts_llm, dim=1)
        else:
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            if self.qformer_text_input:
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            # truncation=True,
            # max_length=self.max_txt_len,
        ).to(image.device)

        empty_targets = torch.ones(atts_llm.size(), dtype=torch.long).to(image.device).fill_(-100)

        # self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'right'
        n_cands = len(candidates)
        with self.maybe_autocast(dtype=torch.bfloat16):
            all_losses = []
            for n in range(n_segments):
                seg_len = n_cands // n_segments
                if n == (n_segments - 1):
                    seg_len = n_cands - seg_len * (n_segments - 1)

                start_i = n * (n_cands // n_segments)
                end_i = start_i + seg_len

                this_output_tokens = self.llm_tokenizer(
                    candidates[start_i:end_i],
                    return_tensors="pt",
                    padding="longest",
                    # truncation=True,
                    # max_length=self.max_output_txt_len,
                ).to(image.device)

                this_input_tokens_ids = text_input_tokens.input_ids.repeat_interleave(seg_len, dim=0)
                this_input_tokens_atts = text_input_tokens.attention_mask.repeat_interleave(seg_len, dim=0)

                this_output_tokens_ids = this_output_tokens.input_ids.repeat(bs, 1)
                this_output_tokens_atts = this_output_tokens.attention_mask.repeat(bs, 1)

                this_llm_tokens, this_input_targets_len = self.concat_text_input_output(
                    this_input_tokens_ids,
                    this_input_tokens_atts,
                    this_output_tokens_ids,
                    this_output_tokens_atts
                )

                this_llm_input_ids = this_llm_tokens['input_ids']
                this_llm_atts = this_llm_tokens['attention_mask']
                # this_llm_input_ids = torch.cat([this_input_tokens_ids, this_output_tokens_ids], dim=1)
                # this_llm_atts = torch.cat([this_input_tokens_atts, this_output_tokens_atts], dim=1)

                inputs_embeds = self.llm_model.get_input_embeddings()(this_llm_input_ids)
                inputs_embeds = torch.cat([inputs_llm.repeat_interleave(seg_len, dim=0), inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm.repeat_interleave(seg_len, dim=0), this_llm_atts], dim=1)

                this_targets = this_llm_input_ids.masked_fill(this_llm_input_ids == self.llm_tokenizer.pad_token_id, -100)
                # this_targets[:, :this_input_tokens_ids.size(1)] = -100
                for i, l in enumerate(this_input_targets_len):
                    this_targets[i][:l] = -100

                this_targets = torch.cat([empty_targets.repeat_interleave(seg_len, dim=0), this_targets], dim=1)

                outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=this_targets,
                    reduction="none",
                )

                loss = outputs.loss

                loss = loss.reshape(bs, seg_len)
                # output_class_ranks = torch.argsort(loss, dim=-1)
                all_losses.append(loss)

            all_losses = torch.cat(all_losses, dim=-1)
            output_class_ranks = torch.argsort(all_losses, dim=-1)

        return output_class_ranks

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llm_model = cfg.get("llm_model")
        memory_bank_length = cfg.get("memory_bank_length", 0)
        num_frames = cfg.get("num_frames", 0)
        max_num_frames = cfg.get("max_num_frames", 120)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 256)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        qformer_text_input = cfg.get("qformer_text_input", True)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            llm_model=llm_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            qformer_text_input=qformer_text_input,
            memory_bank_length=memory_bank_length,
            num_frames=num_frames,
            max_num_frames=max_num_frames,
        )

        # if qformer_text_input:
        #     # Hard-coded to load from BLIP-2 stage-1 pre-trained model (not ideal)
        #     model.load_from_pretrained(
        #         url_or_filename="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth"
        #     )

        model.load_checkpoint_from_config(cfg)

        return model