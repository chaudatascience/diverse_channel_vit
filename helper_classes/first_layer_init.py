from utils import ExtendedEnum


class FirstLayerInit(str, ExtendedEnum):
    ## make attribute capital
    REINIT_AS_RANDOM = "reinit_as_random"
    PRETRAINED_PAD_AVG = "pretrained_pad_avg"  # pad with avg of pretrained weights for additional channels
    PRETRAINED_PAD_RANDOM = "pretrained_pad_random"  # pad with random values for additional channels
    PRETRAINED_PAD_DUPS = "pretrained_pad_dups"  # pad with duplicates channels for additional channels


class NewChannelLeaveOneOut(str, ExtendedEnum):
    ## make attribute capital
    AVG_2 = "avg_2"  ## avg of 2 existing channels
    REPLICATE = "replicate"  ## replicate 1 existing channel
    AVG_2_NOT_IN_CHUNK = "avg_2_not_in_chunk"  ## avg of 2 existing channels not in chunk
    AVG_3 = "avg_3"  ## avg of 3 existing channels
    AVG_3_NOT_IN_CHUNK = "avg_3_not_in_chunk"  ## avg of 3 existing channels not in chunk
    ZERO = "zero"  ##  set zero for the new channels
    IGNORE = "ignore"  ## ignore the new channels in the input images
    AS_IS = "as_is"  ## use with the z_emb it has for hyper channel Vit
    RANDOM = "random"  ## use with the z_emb it has for hyper channel Vit
    EIGENVALUES = "eigenvalues"  ##  img channel -> eigenvalues of the img channel (real part only)
    SIM = "sim"
    DYNAMIC_INPUT_CORR_1 = "dynamic_input_corr_1"
    DYNAMIC_INPUT_CORR_2 = "dynamic_input_corr_2"
    DYNAMIC_INPUT_CORR_3 = "dynamic_input_corr_3"
    DYNAMIC_INPUT_CORR_4 = "dynamic_input_corr_4"
    DYNAMIC_INPUT_CORR_5 = "dynamic_input_corr_5"
    DYNAMIC_INPUT_CORR_6 = "dynamic_input_corr_6"
    FIXED_INPUT_CORR = "fixed_input_corr"
    RANDOM_INPUT_CORR = "random_input_corr"
    GENERATED = "generated"
