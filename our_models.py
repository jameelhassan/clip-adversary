import clip
import torch
from torch import nn
from einops import rearrange, repeat


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


# class FrozenCLIPEmbedder(AbstractEncoder):
#     """Uses the CLIP transformer encoder for text"""
#     def __init__(self, version="ViT-B/32", device="cuda",):
#         super().__init__()
#         self.tokenizer = clip.tokenize
#         self.transformer = clip.load(version)
#         self.device = device
#         self.freeze()

#     def freeze(self):
#         self.transformer = self.transformer.eval()
#         for param in self.parameters():
#             param.requires_grad = False

#     def forward(self, text):
#         tokens = self.tokenizer(text).cuda()
#         outputs = self.transformer.encode_text(tokens).float()
#         outputs /= outputs.norm(dim=-1, keepdim=True)
#         z = outputs

#         return z 

#     def encode(self, text):
#         return self(text)


class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """
    def __init__(self, version='ViT-B/32', device="cuda", max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device="cpu")
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat    #Whats this?
        self.normalize = normalize

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim==2:
            z = z[:, None, :]
        # Huh???
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
        return z


class FrozenClipImageEmbedder(nn.Module):
    """
        Uses the CLIP image encoder.
        """
    def __init__(
            self,
            model,
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
        ):
        super().__init__()
        self.model, self.preprocess = clip.load(name=model, device=device, jit=jit)

        # self.antialias = antialias
        # self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        # self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]
        # x = kornia.geometry.resize(x, (224, 224),
        #                            interpolation='bicubic',align_corners=True,
        #                            antialias=self.antialias)
        # x = (x + 1.) / 2.
        # # renormalize according to clip
        # x = kornia.enhance.normalize(x, self.mean, self.std)
        x = self.preprocess(x)

        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        z_img = self.model.encode_image(self.preprocess(x))
        z_img /= z_img.norm(dim=-1, keepdim=True)
        
        return z_img 


class UNetGenerator():
    
    pass


