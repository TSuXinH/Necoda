import torch
import torch.nn as nn
from model_box.pass_way import PassWay, SpatiotemporalCodeLayer
from model_box.gap_module import GapModule, get_gap_module


class NeRPSTModeInterp(nn.Module):
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
        act=nn.GELU,
        interp_method='interp'
    ):
        super().__init__()
        self.interp_encoder = get_gap_module(
            raw_size_x,
            raw_size_t,
            interp_size_x,
            interp_size_t,
            1,
            interp_chn,
            do_ds=True,
            interp_method=interp_method
        )
        self.interp_decoder = get_gap_module(
            raw_size_x,
            raw_size_t,
            interp_size_x,
            interp_size_t,
            interp_chn,
            1,
            do_ds=False,
            interp_method=interp_method
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
        # self.head = nn.Conv3d(
        #     chns_list[0],
        #     1,
        #     kernel_size=(5, 5, 5),
        #     stride=(1, 1, 1),
        #     padding=(2, 2, 2)
        # )

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
        # x = self.head(x)
        return x, emb_s, emb_t


class NeRPSTModeInterpDecoder(nn.Module):
    def __init__(
        self,
        nerp_st_pro,
    ):
        super().__init__()
        self.s_decoder = nerp_st_pro.s_pass_way.decoder
        self.t_decoder = nerp_st_pro.t_pass_way.decoder
        self.st_decoder = nerp_st_pro.st_decoder
        self.interp_decoder = nerp_st_pro.interp_decoder
        # self.head = nerp_st.head

    def forward(self, emb_s, emb_t):
        x_s = emb_s
        x_t = emb_t
        for layer in self.s_decoder:
            x_s, _ = layer(x_s)
        for layer in self.t_decoder:
            x_t, _ = layer(x_t)
        x = torch.cat([x_s, x_t], dim=1)
        x, _ = self.st_decoder(x)
        x = self.interp_decoder(x)
        # x = self.head(x)
        return x
