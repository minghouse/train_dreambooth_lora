# https://github.com/diStyApps/Safe-and-Stable-Ckpt2Safetensors-Conversion-Tool-GUI/blob/844880ce083d70a2e949aa1990f6151317faef41/run_app_gui.py#L9

import torch 
from safetensors.torch import save_file
import hashlib
HASH_START = 0x100000
HASH_LENGTH = 0x10000

def get_file_hash(filename):
    with open(filename, "rb") as file:
        m = hashlib.sha256()
        file.seek(HASH_START)
        m.update(file.read(HASH_LENGTH))
        return m.hexdigest()[0:8]
		
def convert_to_st(checkpoint_path, output):
	model_hash = get_file_hash(checkpoint_path)
	print(f'load_weights {checkpoint_path} [{model_hash}].')
	weights = load_weights(checkpoint_path)
	print(f'save_file {checkpoint_path} [{model_hash}].')
	try:
		print(f'Saving {output} [{get_file_hash(output)}].')
		save_file(weights, output)
		print(f'END')
	except Exception as e:
		print(f'Error: {e}')  
	
def load_weights(checkpoint_path):
	try:
		# Load the weights from the checkpoint file, without computing gradients
		with torch.no_grad():
			weights = torch.load(checkpoint_path, map_location=torch.device('cpu'))
			# Check if the weights are contained in a "state_dict" key
			if "state_dict" in weights:
				weights = weights["state_dict"]
				# If the weights are nested in another "state_dict" key, remove it
				if "state_dict" in weights:
					weights.pop("state_dict")
			return weights
			
	except Exception as e:
		print(f'Error: {e}')   
		
convert_to_st("anything-v4.5-vae-swapped.ckpt", "anything-v4.5-vae-swapped.safetensors")