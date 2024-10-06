from utils import ExtendedEnum


class ChannelInitialization(str, ExtendedEnum):
    ## currently only used for Attention Pooling
    ZERO = "zero"
    RANDOM = "random"
