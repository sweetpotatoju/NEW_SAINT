import torch
from torch import nn
import torch.nn.functional as F


class simple_MLP(nn.Module):
    def __init__(self, layers):
        super(simple_MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SAINT(nn.Module):
    def __init__(
            self,
            *,
            num_continuous,
            dim,
            depth,
            heads,
            dim_head=16,
            dim_out=1,
            mlp_hidden_mults=(4, 2),
            mlp_act=None,
            num_special_tokens=0,
            attn_dropout=0.,
            ff_dropout=0.,
            cont_embeddings='MLP',
            scalingfactor=10,
            attentiontype='col',
            final_mlp_style='common',
            y_dim=2
    ):
        super().__init__()

        self.num_continuous = num_continuous
        self.dim = dim
        self.cont_embeddings = cont_embeddings
        self.final_mlp_style = final_mlp_style

        if self.cont_embeddings == 'MLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1, 100, self.dim]) for _ in range(self.num_continuous)])
            input_size = (dim * num_continuous)
            nfeats = num_continuous
        elif self.cont_embeddings == 'pos_singleMLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1, 100, self.dim]) for _ in range(1)])
            input_size = (dim * num_continuous)
            nfeats = num_continuous
        else:
            print('Continuous features are not passed through attention')
            input_size = num_continuous
            nfeats = num_continuous

        self.fc = nn.Linear(input_size, dim)

        l = input_size // 8
        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [dim, *hidden_dimensions, dim_out]
        self.mlp = simple_MLP(all_dimensions)

        if self.final_mlp_style == 'common':
            self.mlp1 = simple_MLP([dim, (num_continuous) * 2, num_continuous])
            self.mlp2 = simple_MLP([dim, (num_continuous), 1])
        else:
            self.mlp1 = simple_MLP([dim, (num_continuous) * 2, num_continuous])
            self.mlp2 = simple_MLP([dim, (num_continuous), 1])

        self.mlpfory = simple_MLP([dim, 1000, y_dim])

    def forward(self, x_cont):
        x_cont_transformed = torch.cat([self.simple_MLP[i](x_cont[:, i:i + 1]) for i in range(self.num_continuous)], dim=-1)
        x_combined = self.fc(x_cont_transformed)
        con_outs = self.mlp2(x_combined)
        return con_outs
