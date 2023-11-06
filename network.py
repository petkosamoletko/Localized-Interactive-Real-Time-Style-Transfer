import torch
import torch.nn as nn
import torchvision

from utils import whitening_coloring
    
class AdaIN(nn.Module):
    '''
    AdaIN module out of the style trasnfer network 
    '''
    def __init__(self):
        super(AdaIN, self).__init__()

    def forward(self, content_feat, style_feat, style_strength = 1.0, eps = 1e-7):
        b, c, h, w = content_feat.size()

        content_mean = content_feat.view(b, c, -1).mean(dim = 2, keepdim=True)
        content_std = content_feat.view(b, c, -1).std(dim = 2, keepdim=True) + eps

        style_mean = style_feat.view(b, c, -1).mean(dim = 2, keepdim=True)
        style_std = style_feat.view(b, c, -1).std(dim = 2, keepdim=True)

        stylized_content = (((content_feat.view(b, c, -1) - content_mean) / content_std) * style_std) + style_mean

        if style_strength >= 1:
            stylized_features = style_strength * stylized_content.view(b, c, h, w)
        else:
            stylized_features = ((1 - style_strength) * content_feat) + (style_strength * stylized_content.view(b, c, h, w))

        return stylized_features

class AdaIN_Encoder(nn.Module):
    '''
    Encoder of the style transfer network
    '''
    def __init__(self):        
        super(AdaIN_Encoder, self).__init__()
        vgg = torchvision.models.vgg19(pretrained=True).features

        layers_list1 = [vgg[i] for i in range(0, 2)]
        self.encoder1 = nn.Sequential(*layers_list1) 

        layers_list2 = [vgg[i] for i in range(2, 7)]
        self.encoder2 = nn.Sequential(*layers_list2)  
        
        layers_list3 = [vgg[i] for i in range(7, 12)]
        self.encoder3 = nn.Sequential(*layers_list3) 

        layers_list4 = [vgg[i] for i in range(12, 21)]
        self.encoder4 = nn.Sequential(*layers_list4)
        
    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        return [x1, x2, x3, x4]
    
class AdaIN_Decoder(nn.Module):
    '''
    Decoder of the style trasnfer network
    '''
    def __init__(self):
        super(AdaIN_Decoder, self).__init__()
        
        kernel = (3, 3)
        stride = (1, 1)
        extend_borders = nn.ReflectionPad2d((1, 1, 1, 1))
        upsample = nn.Upsample(scale_factor= 2)
        relu = nn.ReLU()
        
        self.decoder4 = nn.Sequential(
            extend_borders, nn.Conv2d(512, 256, kernel, stride), relu, upsample,
            extend_borders, nn.Conv2d(256, 256, kernel, stride), relu,
            extend_borders, nn.Conv2d(256, 256, kernel, stride), relu,
            extend_borders, nn.Conv2d(256, 256, kernel, stride), relu)
        
        self.decoder3 = nn.Sequential(
            extend_borders, nn.Conv2d(256, 128, kernel, stride), relu, upsample,
            extend_borders, nn.Conv2d(128, 128, kernel, stride), relu)
        
        self.decoder2 = nn.Sequential(
            extend_borders, nn.Conv2d(128, 64, kernel, stride), relu, upsample,
            extend_borders, nn.Conv2d(64, 64, kernel, stride), relu)
        
        self.decoder1 = nn.Sequential(
            extend_borders, nn.Conv2d(64, 3, kernel, stride)) # here we are dropping the ReLU layer as it is the last decoder piece
        
    def forward(self, x):
        x = self.decoder4(x)
        x = self.decoder3(x)
        x = self.decoder2(x)
        x = self.decoder1(x)

        return x
    
class modified_AdaIN_network(nn.Module):
    '''
    Forward pass of the style trasnfer network 
    '''
    def __init__(self):
        super(modified_AdaIN_network, self).__init__()        
        self.encoder = AdaIN_Encoder()
        self.decoder = AdaIN_Decoder()        
        self.adain = AdaIN()

    def forward(self, content_image, styles_images, style_strength, mask_weight, masks, keep_content_color = False):
        # Extract features from content and style images
        content_feature = self.encoder(content_image)
        style_features = [
            self.encoder(whitening_coloring(style, content_image) if keep_content_color else style)
            for style in styles_images]

        # Process masks if they are torch.Tensor
        processed_masks = [
            torch.nn.functional.interpolate(mask, size=content_feature[-1].size()[-2:])
            if isinstance(mask, torch.Tensor) else mask
            for mask in masks
        ]
        # shape check for troubleshooting
        #print("Mask.shapes (individual)", [mask.shape for mask in processed_masks])

        # Compute stylized features based on masks
        post_adain_resultant_features = [
            (self.adain(content_feature[-1], style_feature[-1], style_strength) * mask) * weight
            for style_feature, weight, mask in zip(style_features, mask_weight, processed_masks)
        ]

        # Compute the union of all provided masks and mask content feature
        combined_mask = torch.sum(torch.stack(processed_masks), dim = 0)
        content_mask = 1 - combined_mask

        # shape check for troubleshooting
        #print("Shape of content_feature[-1]:", content_feature[-1].shape)
        #print("Shape of content_mask:", content_mask.shape)

        content_masked_feature = content_feature[-1] * content_mask
        post_adain_resultant_features.append(content_masked_feature)

        # Combine features and decode to get stylized image
        combined_features = sum(post_adain_resultant_features)
        stylized_image = self.decoder(combined_features)

        return stylized_image