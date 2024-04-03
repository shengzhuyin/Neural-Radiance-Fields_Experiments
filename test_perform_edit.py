import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

def perform_edit(original_image, alpha, type = "ZOOM"):
    assert type in ["COLOR", "ZOOM"], f"{type = }"
    if type == "COLOR":
        assert alpha.shape[0] == 3, f"{alpha.shape = }"
        assert len(original_image.shape) == 2 and original_image.shape[1] == 3, f"{original_image.shape = }"
        
        # convert (HxW, 3) to (H, W, 3)
        img_size = int(np.sqrt(original_image.shape[0]))
        image = original_image.view(img_size, img_size, 3)
        #normalize image
        max_, _ = torch.max(image, dim = -1, keepdim = True)
        image = image / (max_ + 1e-5)
        
        # do the edit by balancing the color channels
        new_image = torch.zeros_like(image)
        new_image = torch.matmul(image, torch.diag(alpha))
        max_, _ = torch.max(new_image, dim = -1, keepdim = True)
        new_image = new_image / (max_ + 1e-5) # normalize
        
        mask = ((torch.min(image, dim=-1)[0]) >= 1-(1e-2))
        mask = mask.unsqueeze(-1).expand(-1, -1, 3)
        # print(f"[DEBUG] {mask = }")
        new_image[mask] = 1.0
        
        new_image = new_image.view(original_image.shape)
        assert new_image.shape == original_image.shape, f"{new_image.shape = }, {original_image.shape = }"
        return new_image
    
    elif type == "ZOOM":
        scale_factor = torch.exp(alpha)    # to allow for positive and negative zoom
        assert alpha.shape[0] == 1, f"{alpha.shape = }"
        assert len(original_image.shape) == 2 and original_image.shape[1] == 3, f"{original_image.shape = }"
        
        img_size = int(np.sqrt(original_image.shape[0]))
        image = original_image.view(img_size, img_size, 3)
        
        new_height = int(img_size * scale_factor)
        new_width = int(img_size * scale_factor)
        np_image = image.detach().cpu().numpy()
        new_image = cv2.resize(np_image, (new_width, new_height), interpolation = cv2.INTER_LINEAR)
        
        if scale_factor > 1:
            crop_start_x = (new_width - img_size) // 2
            crop_start_y = (new_height - img_size) // 2
            crop_end_x = crop_start_x + img_size
            crop_end_y = crop_start_y + img_size
            new_image = new_image[crop_start_y:crop_end_y, crop_start_x:crop_end_x]
        
        elif scale_factor < 1:
            pad_height = max(0, img_size - new_height)
            pad_width = max(0, img_size - new_width)

            # Calculate the padding on each side
            top_pad = pad_height // 2
            left_pad = pad_width // 2
            bottom_pad = pad_height - top_pad
            right_pad = pad_width - left_pad
            
            new_image = cv2.copyMakeBorder(
                new_image,
                top_pad,
                bottom_pad,
                left_pad,
                right_pad,
                cv2.BORDER_CONSTANT,
                value=[1, 1, 1]  # You can specify the padding color here (black in this case)
            )
        
        new_image = torch.tensor(new_image).to(image.device)
        max_, _ = torch.max(new_image, dim = -1, keepdim = True)
        new_image = new_image / (max_ + 1e-5) # normalize
        
        new_image = new_image.view(original_image.shape)
        assert new_image.shape == original_image.shape, f"{new_image.shape = }, {original_image.shape = }"
        return new_image

ones = torch.ones(size = (32,32))
halves = torch.ones(size = (32,32)) * 0.5
zeros = torch.zeros(size = (32,32))

path = f"logs/photoshapes/interpolation_INTERPO_COLOR_TEST/0000_rgb.png"
#load image in greyscale and convert to tensor
image = cv2.imread(path, cv2.IMREAD_COLOR)
image = torch.tensor(image).float() / 255.0
print(f"{image.shape = }")

plt.imshow(image)
plt.savefig("original.png")
plt.close()

def show_flat_image(image, p):
    n = int(np.sqrt(image.shape[0]))
    image = image.reshape(n,n,3)
    image = image.detach().cpu().numpy()
    plt.imshow(image)
    plt.savefig(p)
    plt.close()

flat_image = image.reshape(-1, 3)
print(f"[DEBUG] {flat_image.shape = }")
show_flat_image(flat_image, "0.png")

# new_image = perform_edit(flat_image, torch.tensor([0.1/2, 1/2, 0]))
new_image = perform_edit(flat_image, torch.tensor([0.5, 1.0, 1.0]), type = "COLOR")

print(f"[DEBUG] {new_image.shape = }")
show_flat_image(new_image, "1.png")
