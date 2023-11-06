#imports
import cv2
import numpy as np
import torch
import torchvision

from torchvision import transforms
from PIL import Image
from PySide6.QtGui import QImage

def change_hsv(img_path, masks, hex_provided_color_codes, mask_keys=None, value_factor=0.4, sat_factor=0):
    '''
    Changes the color distrubition of a masked object, 
    prior to feeding it into the style transfer network of AdaIN
    '''
    if mask_keys is None:
        mask_keys = masks.keys() 

    # Load content image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert image format from RGB format to HSV, so that specifically the 
    # Hue can be targeted, to achieve more realistic style trasnfer
    hsv_format_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    for key in mask_keys:
        mask = masks[key]
        hex_provided_color_code = hex_provided_color_codes[key]
        
        if hex_provided_color_code is None:
            continue 
        
        # Alter color palette according to the provided referance hex color code 
        r, g, b = tuple(int(hex_provided_color_code[i:i+2], 16) for i in (1, 3, 5))
        target_rgb = np.array([r, g, b]) / 255.0

        target_hsv = cv2.cvtColor(np.uint8([[target_rgb * 255]]), cv2.COLOR_RGB2HSV)[0,0]

        # Additional Value and Saturation changes, from the DoubleSpinBoxes in the GUI
        hsv_format_img[mask, 0] = target_hsv[0]
        sat = target_hsv[1] / 255.0
        new_sat = hsv_format_img[mask, 1] * (sat_factor + (1 - sat_factor) * sat)
        hsv_format_img[mask, 1] = np.clip(new_sat, 0, 255)

        val = target_hsv[2] / 255.0
        new_val = hsv_format_img[mask, 2] * (value_factor + (1 - value_factor) * val)
        hsv_format_img[mask, 2] = np.clip(new_val, 0, 255)

    rgb_format_image = cv2.cvtColor(hsv_format_img, cv2.COLOR_HSV2RGB)

    return rgb_format_image


def tensor_adapter(target_size=None):
    '''
    Adapt an image to a tesnor format, a desired size and 
    normalized state
    '''
    move_to_tensor_format = transforms.ToTensor()
    image_net_normalization = transforms.Normalize([0.485, 0.456, 0.406], 
                                                   [0.229, 0.224, 0.225])

    resize_transform = [transforms.Resize(target_size)] if target_size else []
    tensor_and_normalize_transform = [move_to_tensor_format, 
                                      image_net_normalization]

    finalized_tensor = transforms.Compose(resize_transform + tensor_and_normalize_transform)
    return finalized_tensor


def content_loader(img_array, target_size=None):
    '''
    Convert content image into a tensor format of desired size 
    '''
    # Convert numpy array to PIL image
    img_pil_format = Image.fromarray(img_array).convert("RGB")
    tensor = tensor_adapter(target_size=target_size)(img_pil_format)
    return tensor.unsqueeze(0)


def style_loader(pixmap, target_size=None):
    '''
    Convert style image into a tensor format of desired size 
    '''
    # Convert QPixmap to PIL Image
    img_pil_format = Image.fromqpixmap(pixmap)
    img_pil_format = img_pil_format.convert("RGB")
    tensor = tensor_adapter(target_size=target_size)(img_pil_format)
    return tensor.unsqueeze(0)


def mask_loader(binary_mask):
    '''
    Convert binary mask into a tensor format
    '''
    mask_array_255 = binary_mask.astype(np.uint8) * 255
    grayscale_mask = Image.fromarray(mask_array_255)
    mask_tensor = transforms.functional.to_tensor(grayscale_mask)
    return mask_tensor.unsqueeze(0)


def stylized_output_converter(image):
    '''
    Convert the resultant post-style trasnfer tensor to an image format
    '''
    if isinstance(image, Image.Image):
        #print("Processing as a PIL Image.")
        image = image.convert("RGBA")
        width, height = image.size
        qimage = QImage(image.tobytes(), width, height, QImage.Format_RGBA8888)

    elif isinstance(image, torch.Tensor):
        #print("Processing as a Torch Tensor.")
        image_tensor = image
        adjusted_tensor = image_tensor.squeeze()
        normalization_transform = transforms.Normalize([-2.118, -2.036, -1.804], 
                                                       [4.367, 4.464, 4.444])
        normalized_tensor = normalization_transform(adjusted_tensor)
        grid_tensor = torchvision.utils.make_grid(normalized_tensor)
        clamped_tensor = grid_tensor.clamp_(0.0, 1.0)
        
        converted_image = transforms.functional.to_pil_image(clamped_tensor)
        image = converted_image.convert("RGBA")
        width, height = image.size
        qimage = QImage(image.tobytes(), width, height, QImage.Format_RGBA8888)

    return qimage 

def whitening(style_feat, eps = 0.00001):
    '''
    Whitening Stage from the WCT method
    '''
    style_c, style_h, style_w = style_feat.size()
    style_2d_mean = torch.mean(style_feat.view(style_c, style_h * style_w), dim=1, keepdim=True)
    style_feat_reshaped = style_feat.view(style_c, style_h*style_w)
    style_zero_mean = style_feat_reshaped - style_2d_mean

    style_conv = torch.mm(style_zero_mean, style_zero_mean.t()) + torch.eye(style_feat.shape[0]).float()
    _, style_s, style_v = torch.svd(style_conv) 
    style_k = style_feat.shape[0]
    for i in range(style_feat.shape[0]):
        if style_s [i] < eps:
            style_k = i
            break

    style_diag = torch.diag(style_s[0:style_k].pow(-0.5))
    first_step = torch.mm(style_v[:, 0:style_k], style_diag)
    second_step = torch.mm(first_step, style_v[:, 0:style_k].t())
    whitened_style = torch.mm(second_step, style_zero_mean)

    whitened_style = whitened_style.view(style_feat.shape)
    
    return whitened_style

def coloring(style_feat, content_feat, eps = 0.00001):
    '''
    Coloring Stage from the WCT method
    '''
    style_c, style_h, style_w = style_feat.size()   

    cont_c, cont_h, cont_w = content_feat.size()
    cont_2d_mean = torch.mean(content_feat.view(cont_c, cont_h * cont_w), dim=1, keepdim=True)
    cont_feat_reshaped = content_feat.view(cont_c, cont_h*cont_w)
    cont_zero_mean = cont_feat_reshaped - cont_2d_mean
        
    cont_conv = torch.mm(cont_zero_mean, cont_zero_mean.t()) + torch.eye(content_feat.shape[0]).float()
    _, cont_s, cont_v = torch.svd(cont_conv)
    cont_k = content_feat.shape[0]
    for i in range(content_feat.shape[0]):
        if cont_s [i] < eps:
            cont_k = i
            break

    cont_diag = torch.diag(cont_s[0:cont_k].pow(0.5))
    first_step = torch.mm(cont_v[:, 0:cont_k], cont_diag)
    second_step = torch.mm(first_step, cont_v[:, 0:cont_k].t())

    whitened_style_colored = torch.mm(second_step, style_feat.view(style_c, style_h * style_w)) + cont_2d_mean
    whitened_style_colored = whitened_style_colored.view(style_feat.shape)

    return whitened_style_colored

def whitening_coloring(style, content):
    '''
    Application of the WCT method to batched formatted data
    '''
    whitened_cont = torch.stack([whitening(s) for s in style])

    colored_cont = torch.stack([coloring(whitened_cont[idx], content[idx]) for idx in range(style.size(0))])

    return colored_cont

def dict_keys_update(state_dict):
    '''
    Dictionary adapting the trained AdaIN pth to the method's 
    utalized encoder and decoder structures
    '''
    encode_decode_map = {
        "encoder.encoder.0.0": "encoder.encoder1.0",
        "encoder.encoder.1.2": "encoder.encoder2.0",
        "encoder.encoder.1.5": "encoder.encoder2.3",
        "encoder.encoder.2.7": "encoder.encoder3.0",
        "encoder.encoder.2.10": "encoder.encoder3.3",
        "encoder.encoder.3.12": "encoder.encoder4.0",
        "encoder.encoder.3.14": "encoder.encoder4.2",
        "encoder.encoder.3.16": "encoder.encoder4.4",
        "encoder.encoder.3.19": "encoder.encoder4.7",
    
        "decoder.decoder.0.1": "decoder.decoder4.1",
        "decoder.decoder.0.5": "decoder.decoder4.5",
        "decoder.decoder.0.8": "decoder.decoder4.8",
        "decoder.decoder.0.11": "decoder.decoder4.11",
        "decoder.decoder.1.14": "decoder.decoder3.1",
        "decoder.decoder.1.18": "decoder.decoder3.5",
        "decoder.decoder.2.21": "decoder.decoder2.1",
        "decoder.decoder.2.25": "decoder.decoder2.5",
        "decoder.decoder.3.28": "decoder.decoder1.1"
    }

    for old_value, new_value in encode_decode_map.items():
        state_dict[new_value + ".weight"] = state_dict.pop(old_value + ".weight")
        state_dict[new_value + ".bias"] = state_dict.pop(old_value + ".bias")
    
    return state_dict
        
    


