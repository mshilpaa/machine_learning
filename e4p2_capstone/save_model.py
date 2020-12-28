

import torch
def save_model_cpu(model,path,device):
	model.to('cpu')
	model.eval()
	traced_model = torch.jit.trace(model,torch.randn(1,3,244,244)) 
	traced_model.save(path)
	model.to(device)
	print(f'Saved Model to {path}')
