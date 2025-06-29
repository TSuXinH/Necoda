import torch
import torch.nn as nn
from model_box.pass_way import PassWay, SpatiotemporalCodeLayer
from model_box.gap_module import GapModule


class NeRPBase(nn.Module):
    def __init__(
        self,
        raw_size_x,
        raw_size_t,
        interp_size_x,
        interp_size_t,
        interp_chn,
        pre_s_rate,
        pre_t_rate,
        embedding_dim,
        s_rate_list,
        t_rate_list,
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
            chns_list[0],
            interp_chn,  # chns_list[0],
            act=act,
            do_ds=False,
        )
        self.pass_way = PassWay(
            embedding_dim,
            s_rate_list,
            t_rate_list,
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

    def forward(self, x, emb=None):
        if emb is None:
            x = self.interp_encoder(x)
            x, _ = self.st_encoder(x)
            x, emb = self.pass_way(x)
        else:
            x = emb
            for layer in self.pass_way.decoder:
                x, _ = layer(x)
        x, _ = self.st_decoder(x)
        x = self.interp_decoder(x)
        return x, emb


class NeRPBaseDecoder(nn.Module):
    def __init__(
        self,
        nerp_base,
    ):
        super().__init__()
        self.decoder = nerp_base.pass_way.decoder
        self.st_decoder = nerp_base.st_decoder
        self.interp_decoder = nerp_base.interp_decoder
        # self.head = nerp_st.head

    def forward(self, emb):
        x = emb
        for layer in self.decoder:
            x, _ = layer(x)
        x, _ = self.st_decoder(x)
        x = self.interp_decoder(x)
        return x
