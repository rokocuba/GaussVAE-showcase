import numpy as np
import torch
from skimage import filters
from torchvision.transforms.functional import resize

from utils.saliency import decoder, resnet


def get_smap(image, path, filter_size=15):
    """
    Compute the saliency map of the target image using EMLNet.
    Reference: https://arxiv.org/abs/1805.01047
    Reference: https://github.com/SenJia/EML-NET-Saliency
    """
    if image.shape[0] != 3:
        raise ValueError("Saliency prediction only supports RGB images")
    sod_res = (480, 640)
    imagenet_model = resnet.resnet50(f"{path}/emlnet/res_imagenet.pth").cuda().eval()
    places_model = resnet.resnet50(f"{path}/emlnet/res_places.pth").cuda().eval()
    decoder_model = decoder.build_decoder(f"{path}/emlnet/res_decoder.pth", sod_res, 5, 5).cuda().eval()
    image_sod = resize(image, sod_res).unsqueeze(0)
    with torch.no_grad():
        imagenet_feat = imagenet_model(image_sod, decode=True)
        places_feat = places_model(image_sod, decode=True)
        smap = decoder_model([imagenet_feat, places_feat])
    smap = resize(smap.squeeze(0).detach().cpu(), image.shape[1:]).squeeze(0)

    def post_process(smap):
        smap = filters.gaussian(smap, filter_size)
        smap -= smap.min()
        smap /= smap.max()
        return smap

    return post_process(smap.numpy()).astype(np.float32)
