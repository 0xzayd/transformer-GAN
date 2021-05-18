import torch
import torch.nn as nn

from src.utils import Block, pixel_upsample


class Generator(nn.Module):
    def __init__(self, options, img_size=224, patch_size=16, in_chans=3, num_classes=10, embed_dim=384, depth=5,
                 num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm):
        super(Generator, self).__init__()
        layer_mask = options.mask.split("_")
        mask_l2 = int(layer_mask[0].strip())
        mask_l3 = int(layer_mask[1].strip())

        self.options = options
        self.ch = embed_dim
        self.bottom_width = self.options.bottom_width
        self.embed_dim = embed_dim = self.options.gf_dim
        self.l1 = nn.Linear(self.options.latent_dim, (self.bottom_width ** 2) * self.embed_dim)
        self.pos_embed_1 = nn.Parameter(torch.zeros(1, self.bottom_width**2, embed_dim))
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, (self.bottom_width*2)**2, embed_dim//4))
        self.pos_embed_3 = nn.Parameter(torch.zeros(1, (self.bottom_width*4)**2, embed_dim//16))
        self.pos_embed = [
            self.pos_embed_1,
            self.pos_embed_2,
            self.pos_embed_3
        ]
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.upsample_blocks = nn.ModuleList([
                 nn.ModuleList([
                    Block(
                        dim=embed_dim//4, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer, is_mask=((self.bottom_width*2, mask_l2))),
                    Block(
                        dim=embed_dim//4, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer, is_mask=((self.bottom_width*2, mask_l2))),
                    Block(
                        dim=embed_dim//4, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer, is_mask=((self.bottom_width*2, mask_l2))),
                    Block(
                        dim=embed_dim//4, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer, is_mask=((self.bottom_width*2, mask_l2))),
                 ]
                ),
                 nn.ModuleList([
                    Block(
                        dim=embed_dim//16, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer, is_mask=((self.bottom_width*4, mask_l3))),
                    Block(
                        dim=embed_dim//16, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer, is_mask=((self.bottom_width*4, mask_l3)))
                 ]
                )
                ])
        for i in range(len(self.pos_embed)):
            trunc_normal_(self.pos_embed[i], std=.02)

        self.to_rgb = nn.Sequential(
            nn.BatchNorm2d(self.options.gf_dim),
            nn.ReLU(),
            nn.Tanh()
        )
        
        self.deconv = nn.Sequential(
            nn.Conv2d(self.embed_dim//16, 3, 1, 1, 0)
        )

    def set_arch(self, x, cur_stage):
        pass

    def forward(self, z, epoch):
        x = self.l1(z).view(-1, self.bottom_width ** 2, self.embed_dim)
        x = x + self.pos_embed[0].to(x.get_device())
        B = x.size()
        H, W = self.bottom_width, self.bottom_width
        for index, blk in enumerate(self.blocks):
            x = blk(x, epoch)
        for index, blk in enumerate(self.upsample_blocks):
            x, H, W = pixel_upsample(x, H, W)
            x = x + self.pos_embed[index+1].to(x.get_device())
            for b in blk:
                x = b(x, epoch)
        output = self.deconv(x.permute(0, 2, 1).view(-1, self.embed_dim//16, H, W))
        return output









