import os
import glob
import h5py
import numpy as np
from PIL import Image

import torch
from torch import nn
from torchvision import models
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from transformers import BertTokenizer, GPT2Tokenizer
from transformers import ViTFeatureExtractor, AutoFeatureExtractor

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

'''
Seed randoms
'''
def seed_everything(seed=27):
    '''
    Set random seed for reproducible experiments
    Inputs: seed number 
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

"""
Feature extractor for vd&vbrm
"""
class FeatureExtractor(nn.Module):
    def __init__(self, patch_size = 4):
        super(FeatureExtractor, self).__init__()
        # visual feature extraction
        self.img_feature_extractor = models.resnet18(pretrained=True)
        self.img_feature_extractor = torch.nn.Sequential(*(list(self.img_feature_extractor.children())[:-2]))
        self.resize_dim = nn.AdaptiveAvgPool2d((patch_size,patch_size))
        
    def forward(self, img):
        outputs = self.resize_dim(self.img_feature_extractor(img))
        return outputs


"""
Evaluation
"""
def eval(model_ver,img,question):
    '''
    Train and test for cholec dataset
    '''
    # GPU or CPU
    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
    cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

    # text_copy of question
    text_input = question

    # tokenizer
    if model_ver == "vb" or model_ver == "vb-rr":
        tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v2/bert-Cholec80-VQA/', do_lower_case=True)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    

    # process img 
    if model_ver == "efvlegpt2ViT": 
        transform = None
        image_processor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    elif model_ver == "efvlegpt2Swin":
        transform = None
        image_processor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    elif model_ver == "efvlegpt2rs18":
        transform = transforms.Compose([
                                    transforms.Resize((300,256)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                    ])
    elif model_ver == "vb" or model_ver == "vb-rr":
        transform = transforms.Compose([transforms.Resize((300,256)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                ])
        image_processor = FeatureExtractor(patch_size = 5).to(device).eval()
        
    if model_ver == "efvlegpt2ViT" or model_ver == "efvlegpt2Swin":
        visual_feature = image_processor(Image.fromarray(img), return_tensors="pt")
        visual_feature['pixel_values'] = torch.squeeze(visual_feature['pixel_values'],1)
    elif model_ver == "efvlegpt2rs18":
        visual_feature = transform(Image.fromarray(img))
        visual_feature = visual_feature.unsqueeze(dim=0).to(device)
    elif model_ver == "vb" or "vb-rr":
        visual_feature = transform(Image.fromarray(img)).to(device)
        visual_feature = image_processor(visual_feature.unsqueeze(dim=0))
        visual_feature = torch.flatten(visual_feature, start_dim=2)
        visual_feature = visual_feature.permute((0,2,1))
        
    # process question
    if model_ver == 'vb' or model_ver == 'vb-rr':
        question = tokenizer([question], return_tensors="pt", padding="max_length",max_length=25)
    elif model_ver == 'efvlegpt2rs18' or model_ver == "efvlegpt2Swin" or model_ver == 'efvlegpt2ViT':
        question = tokenizer([question], padding="max_length",max_length=25, return_tensors="pt")

    # num_classes
    num_class = 13
    
    # pre-trained model
    if model_ver == "efvlegpt2Swin":
        checkpoint = torch.load("checkpoints/efvlegpt2Swin/c80/v1_z_qf_Best.pth.tar",map_location=str(device))
    elif model_ver == "efvlegpt2ViT":
        checkpoint = torch.load("checkpoints/efvlegpt2ViT/c80/v1_z_qf_Best.pth.tar",map_location=str(device))
    elif model_ver == "efvlegpt2rs18":
        checkpoint = torch.load("checkpoints/efvlegpt2rs18/c80/v3_z_qf_Best.pth.tar",map_location=str(device))
    elif model_ver == "vb":
        checkpoint = torch.load("checkpoints/vb/c80/vb_Best.pth.tar",map_location=str(device))
    elif model_ver == "vb-rr":
        checkpoint = torch.load("checkpoints/vbrm/c80/vbrm_Best.pth.tar",map_location=str(device))
    model = checkpoint['model']

    # labelsAttributeError: Can't get attribute 'PatchEmbeddings' on <module 'transformers.models.vit.modeling_vit'
    labels = ['no', 'calot triangle dissection', 'yes', '1', '2', 'gallbladder dissection', 
                        'clipping cutting', 'gallbladder retraction', '0', 'cleaning coagulation', 
                        'gallbladder packaging', 'preparation', '3']
    # Move to GPU, if available
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        # ans = question['input_ids']
        outputs = model(question, visual_feature)
        # ans = outputs.detach().cpu().numpy()
        ans = labels[int(outputs.detach().cpu().numpy().argmax())]
    
    # langchain chatgpt_description
    with open("api.txt") as f:
        api_key = f.readline()
    llm = OpenAI(openai_api_key=api_key,temperature=.1)
    template = """You are an AI assistant specializing in surgical tasks. Based on the Question and Answer provided you will have to provide a helpful, harmless, and honest description.
                  Question :{question}
                  Answer : {answer}
                  Description : 
    """
    prompt_template = PromptTemplate(input_variables=["question","answer"], template=template)
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    description = llm_chain.run(question=text_input,answer=ans)
    
    return ans,description

