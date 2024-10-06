import numpy as np
from einops import rearrange, repeat
from timm import create_model
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import random
from config import Model
from helper_classes.feature_pooling import FeaturePooling


class TemplateMixingConvNeXt(nn.Module):
    def __init__(
        self,
        config: Model,
        **kwargs,
    ):
        # pretrained_model_name "convnext_tiny.fb_in22k"
        ## forward pass: https://github.com/huggingface/pytorch-image-models/blob/b3e816d6d71ec132b39c603d68b619ae2870fd0a/timm/models/convnext.py#L420

        super().__init__()
        self.cfg = config
        self.enable_sample = config.enable_sample
        model = create_model(config.pretrained_model_name, pretrained=config.pretrained)
        mapper = kwargs["mapper"]
        self.mapper = mapper
        num_templates = len(config.in_channel_names) * config.num_templates_per_channel

        out_dim, original_in_dim, kh, kw = model.stem[0].weight.shape
        self.stride = model.stem[0].stride
        self.padding = model.stem[0].padding
        self.dilation = model.stem[0].dilation
        self.groups = model.stem[0].groups

        # First conv layer
        self.conv1_param_bank = nn.Parameter(torch.zeros(out_dim, num_templates, kh, kw))
        self.add_prefix = lambda x: f"chunk_{x}"
        if self.cfg.separate_coef:
            self.conv1_coefs = nn.ParameterDict(
                {
                    self.add_prefix(data_channel): nn.Parameter(torch.zeros(len(channels), num_templates))
                    for data_channel, channels in self.mapper.items()
                }
            )
        else:
            self.conv1_coefs = nn.Parameter(torch.zeros(len(config.in_channel_names), num_templates))

        nn.init.kaiming_normal_(self.conv1_param_bank, mode="fan_in", nonlinearity="relu")
        if isinstance(self.conv1_coefs, nn.ParameterDict):
            for param in self.conv1_coefs.values():
                nn.init.orthogonal_(param)
        else:
            nn.init.orthogonal_(self.conv1_coefs)

        ## Make a list to store reference for easy access later on
        self.adaptive_interface = nn.ParameterList([self.conv1_param_bank, self.conv1_coefs])

        if config.is_conv_small:
            num_stages_1 = 8
            num_stages_2 = 2
        else:
            num_stages_1 = 9
            num_stages_2 = 3

        ## shared feature_extractor
        self.feature_extractor = nn.Sequential(
            # model.stem[1],
            model.stages[0],
            model.stages[1],
            model.stages[2].downsample,
            *[model.stages[2].blocks[i] for i in range(num_stages_1)],
            model.stages[3].downsample,
            *[model.stages[3].blocks[i] for i in range(num_stages_2)],
        )

        self.classifer_head = None
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

    def mix_templates_first_layer(self, chunk: str) -> Tensor:
        """
        @return: return a tensor, shape (out_channels, in_channels, kernel_h, kernel_w)
        """
        assert chunk in self.mapper, f"Invalid chunk: {chunk}"
        if self.cfg.separate_coef:
            coefs = self.conv1_coefs[self.add_prefix(chunk)]
        else:
            coefs = self.conv1_coefs[self.mapper[chunk]]

        coefs = rearrange(coefs, "c t ->1 c t 1 1")
        templates = repeat(self.conv1_param_bank, "o t h w -> o c t h w", c=len(self.mapper[chunk]))
        params = torch.sum(coefs * templates, dim=2)
        return params

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
        new_channel_init=None,
        **kwargs,
    ) -> torch.Tensor:

        b, Cin, h, w = x.shape
        cur_channels = self.mapper[chunk_name]  ## type: ignore
        conv1_params = self.mix_templates_first_layer(
            chunk_name
        )  ##  (out_channels, in_channels, kernel_h, kernel_w)

        if self.training and self.enable_sample:
            Cin_new = random.randint(1, Cin)
            cur_channels = random.sample(cur_channels, k=Cin_new)
            Cin = Cin_new
            channels_idx = [self.mapper[chunk_name].index(c) for c in cur_channels]
            x = x[:, channels_idx, :, :]
            conv1_params = conv1_params[:, channels_idx, :, :]
            # print("x.shape", x.shape)

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

        out = x
        if self.classifer_head is not None:
            out = self.classifer_head(out)
        return out


def templatemixingconvnext(cfg: Model, **kwargs) -> TemplateMixingConvNeXt:
    return TemplateMixingConvNeXt(config=cfg, **kwargs)
