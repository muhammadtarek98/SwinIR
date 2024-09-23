import torch,cv2,torchvision
import numpy as np
from models.network_swinir import SwinIR
def load_model(ckpt_dir,device,model):
  model=model
  state_dict = torch.load(f=ckpt_dir, map_location=device)
  model.load_state_dict(state_dict=state_dict["params"])
  model.to(device)
  model.eval()
  return model
def image_to_tensor(image,factor=8)->torch.Tensor:
  transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Resize((480,480)),
     torchvision.transforms.Normalize(mean=[0, 0, 0],std=[1, 1, 1])
    ]
  )
  image = transform(image)
  image=image.to(torch.float32)
  image=image.unsqueeze(0)
  h, w = image.shape[2], image.shape[3]
  H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
  padh = H - h if h % factor != 0 else 0
  padw = W - w if w % factor != 0 else 0
  image=torch.nn.functional.pad(image, (0,padw,0,padh), 'reflect')
  return image,h,w,H,W

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ckpt_dir="Deraining/pretrained_models/deraining.pth"
model=load_model(ckpt_dir,device,model=SwinIR(upscale=1, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv'))
image=cv2.imread("/home/muahmmad/projects/Image_enhancement/Enhancement_Dataset/9907_no_fish_f000710.jpg")
#image=cv2.cvtColor(src=image,code=cv2.COLOR_BGR2RGB)
tensor,h,w,H,W=image_to_tensor(image)
tensor=tensor.to(device)

print(tensor)
with torch.no_grad():
  pred=model(tensor)
pred = pred[:, :, :h, :w]
pred= torch.clamp(pred,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
pred = (pred * 255).clip(0, 255).astype(np.uint8)
print(pred)
cv2.imshow(mat=pred,winname="image")
cv2.waitKey(0)
cv2.destroyAllWindows()