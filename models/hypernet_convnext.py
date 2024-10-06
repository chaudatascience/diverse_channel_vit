import numpy as np
from einops import rearrange
from timm import create_model
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from models.hypernet import HyperNetwork
from config import Model
from helper_classes.feature_pooling import FeaturePooling
from helper_classes.first_layer_init import NewChannelLeaveOneOut

def get_mapper():
    allen = ["nucleus", "membrane", "protein"]
    hpa = ["microtubules", "protein", "nucleus", "er"]
    cp = ["nucleus", "er", "rna", "golgi", "mito"]
    total = list(sorted(set(allen + hpa + cp)))
    total_dict = {x: i for i, x in enumerate(total)}

    a = [total_dict[x] for x in allen]
    h = [total_dict[x] for x in hpa]
    c = [total_dict[x] for x in cp]
    ## a,h,c: [5, 2, 6], [3, 6, 5, 0], [5, 0, 7, 1, 4]
    return a, h, c


class HyperConvNeXt(nn.Module):
    def __init__(self, config: Model, **kwargs):
        # pretrained_model_name "convnext_tiny.fb_in22k"
        ## forward pass: https://github.com/huggingface/pytorch-image-models/blob/b3e816d6d71ec132b39c603d68b619ae2870fd0a/timm/models/convnext.py#L420

        ## model card: https://huggingface.co/timm/convnext_tiny.fb_in22k
        super().__init__()
        self.cfg = config
        model = create_model(
            config.pretrained_model_name,
            pretrained=config.pretrained,
            # checkpoint_path="/projectnb/ivc-ml/chaupham/pretrained_models",
        )  # type: ignore

        total_in_channels = len(config.in_channel_names)  # type: ignore

        ## all channels in this order (alphabet): ['er', 'golgi', 'membrane', 'microtubules','mito','nucleus','protein', 'rna']
        # self.mapper = {
        #     "Allen": [5, 2, 6],
        #     "HPA": [3, 6, 5, 0],
        #     "CP": [5, 0, 7, 1, 4],
        # }
        mapper = kwargs["mapper"]
        self.mapper = mapper
        out_dim, original_in_dim, kh, kw = model.stem[0].weight.shape
        self.stride = model.stem[0].stride
        self.padding = model.stem[0].padding
        self.dilation = model.stem[0].dilation
        self.groups = model.stem[0].groups

        # First conv layer
        if config.separate_emb:
            self.conv1_emb = nn.ParameterDict(
                {
                    data_channel: torch.randn(len(channels), config.z_dim)
                    for data_channel, channels in self.mapper.items()
                }
            )
        else:
            self.conv1_emb = nn.Embedding(total_in_channels, config.z_dim)

        self.hypernet = HyperNetwork(config.z_dim, config.hidden_dim, kh, out_dim, 1)

        ## shared feature_extractor
        if config.is_conv_small:
            num_stages_1 = 8
            num_stages_2 = 2
        else:
            num_stages_1 = 9
            num_stages_2 = 3
        self.feature_extractor = nn.Sequential(
            model.stem[1],  ## norm_layer(dims[0])
            model.stages[0],
            model.stages[1],
            model.stages[2].downsample,
            *[model.stages[2].blocks[i] for i in range(num_stages_1)],
            model.stages[3].downsample,
            *[model.stages[3].blocks[i] for i in range(num_stages_2)],
        )
        self.classifer_head = nn.Identity()
        if "Allen" not in mapper:  ## if not Morphem dataset
            ## append an classifier layer to the model
            self.classifer_head = nn.Linear(model.num_features, config.num_classes)

        num_proxies = config.num_classes  ## depends on the number of classes of the dataset
        self.dim = 768 if self.cfg.pooling in ["avg", "max", "avgmax"] else 7 * 7 * 768
        self.proxies = torch.nn.Parameter((torch.randn(num_proxies, self.dim) / 8))
        init_temperature = config.temperature  # scale = sqrt(1/T)
        if self.cfg.learnable_temp:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / init_temperature))
        else:
            self.scale = np.sqrt(1.0 / init_temperature)

    def generate_params_first_layer(
        self,
        chunk: str,
        training_chunks_str: str | None,
        new_channel_init: NewChannelLeaveOneOut | None,
    ) -> Tensor:
        assert chunk in self.mapper, f"Invalid chunk: {chunk}"
        if training_chunks_str is not None:  ## leave one out
            training_chunks = training_chunks_str.split("_")
            training_channels = [self.mapper[ch] for ch in training_chunks]
            training_channels = [item for sublist in training_channels for item in sublist]  ## flatten
            chs_not_seen = [c for c in training_channels if c not in self.mapper[chunk]]

        if self.cfg.separate_emb:
            if training_chunks_str is None or chunk in training_chunks:
                z_emb = self.conv1_emb[chunk]
            elif training_chunks_str is not None:  ## leave one out
                z_emb_list = []
                for ch in self.mapper[chunk]:
                    emb_ch = []
                    for dataset, data_channels in self.mapper.items():
                        if ch in data_channels:
                            idx = data_channels.index(ch)
                            emb_ch.append(self.conv1_emb[dataset][idx])
                        else:
                            emb_ch.append(self.conv1_emb[dataset].mean(dim=0))
                    z_emb_list.append(torch.stack(emb_ch).mean(dim=0))
                z_emb = torch.stack(z_emb_list, dim=0)
        else:
            if training_chunks_str is None or self.training:
                z_emb = self.conv1_emb(
                    torch.tensor(self.mapper[chunk], dtype=torch.long, device=self.conv1_emb.weight.device)
                )
            else:
                param_list = []
                cur = 0
                avg2, avg2_not_seen = (
                    NewChannelLeaveOneOut.AVG_2,
                    NewChannelLeaveOneOut.AVG_2_NOT_IN_CHUNK,
                )

                avg3, avg3_not_seen = (
                    NewChannelLeaveOneOut.AVG_3,
                    NewChannelLeaveOneOut.AVG_3_NOT_IN_CHUNK,
                )
                ch_banks = chs_not_seen if "not_in_chunk" in new_channel_init else training_channels
                device = self.conv1_emb.weight.device

                for c in self.mapper[chunk]:
                    if c not in training_channels:
                        if new_channel_init in [avg2, avg2_not_seen]:
                            c1 = ch_banks[cur]
                            c2 = ch_banks[(cur + 1) % len(ch_banks)]
                            param = self.conv1_emb(
                                torch.tensor([c1, c2], dtype=torch.long, device=device)
                            ).mean(dim=0, keepdim=True)
                        elif new_channel_init in [avg3, avg3_not_seen]:
                            c1 = ch_banks[cur]
                            c2 = ch_banks[(cur + 1) % len(ch_banks)]
                            c3 = ch_banks[(cur + 2) % len(ch_banks)]
                            param = self.conv1_emb(
                                torch.tensor([c1, c2, c3], dtype=torch.long, device=device)
                            ).mean(dim=0, keepdim=True)
                        elif new_channel_init == NewChannelLeaveOneOut.REPLICATE:
                            c = ch_banks[cur]
                            param = self.conv1_emb(torch.tensor([c], dtype=torch.long, device=device))
                        elif new_channel_init == NewChannelLeaveOneOut.ZERO:
                            param = torch.zeros_like(self.conv1_emb.weight[0:1])
                        elif new_channel_init == NewChannelLeaveOneOut.RANDOM:
                            param = self.conv1_emb(torch.tensor([c], dtype=torch.long, device=device))
                        else:
                            raise ValueError(f"Invalid new_channel_init: '{new_channel_init}'")
                        cur = (cur + 1) % len(ch_banks)
                    else:
                        param = self.conv1_emb(
                            torch.tensor(
                                [c],
                                dtype=torch.long,
                                device=self.conv1_emb.weight.device,
                            )
                        )
                    param_list.append(param)

                z_emb = torch.cat(param_list, dim=0)
        kernels = self.hypernet(z_emb)
        return kernels

    def _reset_params(self, model):
        for m in model.children():
            if len(list(m.children())) > 0:
                self._reset_params(m)

            elif isinstance(m, nn.Conv2d):
                print("resetting", m)
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                print("resetting", m)

            elif isinstance(m, nn.Linear):
                print("resetting", m)

                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
            else:
                print("skipped", m)

    def _init_bias(self, model):
        ## Init bias of the first layer
        if model.stem[0].bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(model.stem[0].weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(model.stem[0].bias, -bound, bound)

    def forward(
        self,
        x: torch.Tensor,
        chunk_name: str,
        training_chunks: str | None = None,
        init_first_layer=None,  # not used
        new_channel_init: NewChannelLeaveOneOut | None = None,
        **kwargs,
    ) -> torch.Tensor:

        conv1_params = self.generate_params_first_layer(
            chunk_name, training_chunks, new_channel_init=new_channel_init
        )

        x = F.conv2d(
            x,
            conv1_params,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        x = self.feature_extractor(x)
        if self.cfg.pooling == FeaturePooling.AVG:
            x = F.adaptive_avg_pool2d(x, (1, 1))
        elif self.cfg.pooling == FeaturePooling.MAX:
            x = F.adaptive_max_pool2d(x, (1, 1))
        elif self.cfg.pooling == FeaturePooling.AVG_MAX:
            x_avg = F.adaptive_avg_pool2d(x, (1, 1))
            x_max = F.adaptive_max_pool2d(x, (1, 1))
            x = torch.cat([x_avg, x_max], dim=1)
        elif self.cfg.pooling == FeaturePooling.NONE:
            pass
        else:
            raise ValueError(f"Pooling {self.cfg.pooling} not supported. Use one of {FeaturePooling.list()}")
        x = rearrange(x, "b c h w -> b (c h w)")
        x = self.classifer_head(x)
        return x


def hyperconvnext(cfg: Model, **kwargs) -> HyperConvNeXt:
    return HyperConvNeXt(config=cfg, **kwargs)
