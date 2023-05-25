# kohya_ss さん、記事ありがとうございます。❤ ❤ https://note.com/kohya_ss/n/nf5893a2e719c 

import torch

def merge_vae(ckpt, vae, output):
  print(f"load checkpoint: {ckpt}")
  model = torch.load(ckpt, map_location="cpu")
  if "state_dict" in model:
    sd = model["state_dict"]
  else:
    sd = model

  full_model = False

  print(f"load VAE: {vae}")
  vae_model = torch.load(vae, map_location="cpu")
  vae_sd = vae_model['state_dict']

  for vae_key in vae_sd:
    if vae_key.startswith("first_stage_model."):
      full_model = True
      break

  for vae_key in vae_sd:
    sd_key = vae_key
    if full_model:
      if not sd_key.startswith("first_stage_model."):
        continue
    else:
      if sd_key not in sd:
        sd_key = "first_stage_model." + sd_key
    if sd_key not in sd:
      print(f"key not exists in model: {vae_key}")
      continue
    sd[sd_key] = vae_sd[vae_key]

  print(f"saving checkpoint to: {output}")
  torch.save(model, output)

merge_vae("anything-v4.5.ckpt", "anything-v4.0.vae.pt", "anything-v4.5-vae-swapped.ckpt")