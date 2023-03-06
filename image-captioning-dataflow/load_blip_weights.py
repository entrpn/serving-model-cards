from BLIP.models.blip import blip_decoder
import torch
blip_state_dict_path = 'blip_state_dict.pth'
torch.save(torch.load('model*_base_caption.pth')['model'], blip_state_dict_path)