import torch
import torch.nn as nn
from model_box.spatiotemporal_code import SpatiotemporalCodeLayer


class PassWay(nn.Module):
    def __init__(
        self,
        embedding_dim,
        s_rate_list,
        t_rate_list,
        chns_list,
        act=nn.GELU
    ):
        super().__init__()
        encoder = []
        decoder = []
        for idx in range(len(s_rate_list)):
            encoder.append(
                SpatiotemporalCodeLayer(
                    s_rate_list[idx],
                    t_rate_list[idx],
                    chns_list[idx],
                    chns_list[idx+1],
                    act,
                    do_ds=True,
                    apply_in=True
                ) if idx < len(s_rate_list)-1
                else SpatiotemporalCodeLayer(
                    s_rate_list[idx],
                    t_rate_list[idx],
                    chns_list[idx],
                    embedding_dim,
                    act,
                    do_ds=True,
                    apply_in=True,
                )
            )
            decoder.append(
                SpatiotemporalCodeLayer(
                    s_rate_list[-(idx+1)],
                    t_rate_list[-(idx+1)],
                    embedding_dim,
                    chns_list[-(idx+1)],
                    do_ds=False,
                    apply_in=True,
                ) if idx == 0
                else SpatiotemporalCodeLayer(
                    s_rate_list[-(idx+1)],
                    t_rate_list[-(idx+1)],
                    chns_list[-idx],
                    chns_list[-(idx+1)],
                    do_ds=False,
                    apply_in=True,
                )
            )
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        for layer in self.encoder:
            x, _ = layer(x)
        emb = x
        for layer in self.decoder:
            x, _ = layer(x)
        return x, emb

