import torch
import torch.nn as nn
from model_box.pass_way import PassWay, SpatiotemporalCodeLayer
from model_box.gap_module import GapModule


class NeRPSTPro(nn.Module):
    def __init__(
        self,
        raw_size_x,
        raw_size_t,
        interp_size_x,
        interp_size_t,
        interp_chn,
        pre_s_rate,
        pre_t_rate,
        s_embedding_dim,
        t_embedding_dim,
        s_s_rate_list,
        s_t_rate_list,
        t_s_rate_list,
        t_t_rate_list,
        chns_list,
        act=nn.GELU
    ):
        super().__init__()
        self.interp_encoder = GapModule(
            raw_size_x,
            raw_size_t,
            1,
            interp_chn,
        )
        self.interp_decoder = GapModule(
            interp_size_x,
            interp_size_t,
            interp_chn,
            1,
            final_act=False,
        )
        self.st_encoder = SpatiotemporalCodeLayer(
            pre_s_rate,
            pre_t_rate,
            interp_chn,
            chns_list[0],
            act=act,
            do_ds=True,
        )
        self.st_decoder = SpatiotemporalCodeLayer(
            pre_s_rate,
            pre_t_rate,
            chns_list[0]*2,
            interp_chn,  # chns_list[0],
            act=act,
            do_ds=False,
        )
        self.s_pass_way = PassWay(
            s_embedding_dim,
            s_s_rate_list,
            s_t_rate_list,
            chns_list,
            act=act,
        )
        self.t_pass_way = PassWay(
            t_embedding_dim,
            t_s_rate_list,
            t_t_rate_list,
            chns_list,
            act=act,
        )

    def forward(self, x, emb_s=None, emb_t=None):
        if emb_s is None or emb_t is None:
            x = self.interp_encoder(x)
            x, _ = self.st_encoder(x)
            x_s, emb_s = self.s_pass_way(x)
            x_t, emb_t = self.t_pass_way(x)
        else:
            x_s = emb_s
            x_t = emb_t
            for layer in self.s_pass_way.decoder:
                x_s, _ = layer(x_s)
            for layer in self.t_pass_way.decoder:
                x_t, _ = layer(x_t)
        x = torch.cat([x_s, x_t], dim=1)
        x, _ = self.st_decoder(x)
        x = self.interp_decoder(x)
        return x, emb_s, emb_t


class NeRPSTProDecoder(nn.Module):
    def __init__(
        self,
        nerp_st_pro,
    ):
        super().__init__()
        self.s_decoder = nerp_st_pro.s_pass_way.decoder
        self.t_decoder = nerp_st_pro.t_pass_way.decoder
        self.st_decoder = nerp_st_pro.st_decoder
        self.interp_decoder = nerp_st_pro.interp_decoder

    def forward(self, emb_s, emb_t, extract_feature=False):
        if extract_feature:
            x_s = emb_s
            x_t = emb_t
            for layer in self.s_decoder:
                x_s, s_inter = layer(x_s)
            for layer in self.t_decoder:
                x_t, t_inter = layer(x_t)
            x = torch.cat([x_s, x_t], dim=1)
            x, final_inter = self.st_decoder(x)
            x = self.interp_decoder(x)
            return x, x_s, x_t, s_inter, t_inter, final_inter
        x_s = emb_s
        x_t = emb_t
        for layer in self.s_decoder:
            x_s, _ = layer(x_s)
        for layer in self.t_decoder:
            x_t, _ = layer(x_t)
        x = torch.cat([x_s, x_t], dim=1)
        x, _ = self.st_decoder(x)
        x = self.interp_decoder(x)
        return x
