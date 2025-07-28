import torch
import torch.nn as nn
import torch.nn.functional as F
import models
from models import register
from utils import make_coord
from models.LFEM  import  local_enhanced_blcok
import numpy as np
from models.LFEM import *

@register('add_mamba_fusion_mri_ets_text')
class add_MRI_text(nn.Module):

    def __init__(self, encoder_spec,fusion_spec,ref_spec, imnet_spec=None, hidden_dim=256):
        super().__init__()
        self.encoder = models.make(encoder_spec)
        self.fusion = models.make(fusion_spec)
        self.ref_text_encoder=models.make(ref_spec)


        self.coef = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.freq = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.inp_out=nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.phase = nn.Linear(2, hidden_dim // 2, bias=False)
        self.local_enhanced_block = local_enhanced_blcok()
        self.imnet = models.make(imnet_spec, args={'in_dim': hidden_dim})




    def gen_feat(self, inp,ref,ker,ref_t):



        self.inp = inp

        # print('inp shape:', self.inp.shape)
        self.feat_coord = make_coord(inp.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(inp.shape[0], 2, *inp.shape[-2:])

        self.feat= self.encoder(inp,ker)
        self.ref_feat, self.ref_loss = self.local_enhanced_block(ref)


        # print(self.feat.shape,self.ref_feat.shape)
        self.feat=self.fusion(self.feat,self.ref_feat)
        # print(self.feat.shape)

        self.coeff = self.coef(self.feat)
        self.freqq = self.freq(self.feat)
        # print('feat shape:', self.feat.shape, self.coeff.shape, self.freqq.shape)
        self.feat = self.inp_out(self.feat)
        self.ref_txt=self.ref_text_encoder(ref_t)
        return self.feat

    def query_rgb(self, coord, cell=None):

        # print("crood shape:",coord.shape) 65532,2
        feat = self.feat
        ref_txt=self.ref_txt
        coef = self.coeff
        freq = self.freqq

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = self.feat_coord

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                # prepare coefficient & frequency
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_coef = F.grid_sample(
                    coef, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_freq = F.grid_sample(
                    freq, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_reft = F.grid_sample(
                    ref_txt, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)


                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]

                # prepare cell
                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]

                # basis generation
                bs, q = coord.shape[:2]
                q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)
                q_freq = torch.mul(q_freq, rel_coord.unsqueeze(-1))
                q_freq = torch.sum(q_freq, dim=-2)
                q_freq += self.phase(rel_cell.view((bs * q, -1))).view(bs, q, -1)
                q_freq = torch.cat((torch.cos(np.pi * q_freq), torch.sin(np.pi * q_freq)), dim=-1)

                inp = torch.mul(q_coef, q_freq)

                inp= torch.mul(inp,q_reft)


                pred = self.imnet(inp.contiguous().view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0];
        areas[0] = areas[3];
        areas[3] = t
        t = areas[1];
        areas[1] = areas[2];
        areas[2] = t

        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        ret += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear', \
                             padding_mode='border', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        return ret

    def forward(self, inp, ref,ker,ref_t, coord, cell ):

        self.gen_feat(inp,ref,ker,ref_t)
        return self.query_rgb(coord, cell),self.ref_loss


