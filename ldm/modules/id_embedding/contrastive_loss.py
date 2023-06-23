import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, meta_dim=768):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.g_net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(meta_dim, meta_dim)
        )

    def forward(self, metas: list):
        """
        :params metas: id_cnt*(N,768)=(1+diff+1+diff)*(N,768)
        """
        id_cnt = len(metas)
        n, dim = metas[0].shape

        metas = torch.stack(metas)  # (id_cnt,N,768)
        metas = metas.permute(1, 0, 2)  # (N,id_cnt,768)
        metas = rearrange(metas, 'n k c -> (n k) c').contiguous()  # (N*id_cnt,768)

        z_feats = self.g_net(metas)  # (N*id_cnt,768)

        z_feats_list = torch.chunk(z_feats, n)  # N*(id_cnt,768)
        nll = 0.

        for i in range(n):
            z_feats = z_feats_list[i]  # (1+diff+1+diff,768)

            # Calculate cosine similarity
            cos_sim = F.cosine_similarity(z_feats[:, None, :], z_feats[None, :, :], dim=-1)  # (B,B)

            # Mask out cosine similarity to itself
            self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
            cos_sim.masked_fill_(self_mask, -9e15)

            # Find positive example -> batch_size//2 away from the original example
            pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)

            # InfoNCE loss
            cos_sim = cos_sim / self.temperature
            nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
            nll = nll.mean()
            # nll += -cos_sim[0, -1] + torch.logsumexp(cos_sim[0], dim=-1, keepdim=True)  # first and last are same
        return nll.mean()


if __name__ == "__main__":

    loss = ContrastiveLoss()
    a = 16 * [torch.randn(2, 768)]
    x = loss(a)
    print(x)
