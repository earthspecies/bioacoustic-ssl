"""Vendored BAT config.

Upstream: https://huggingface.co/lrauch/BAT-vit-b16-pretrainedAS2M/blob/main/configuration_bat.py
Revision: 175109327540c72f4b678b149e7cfaf0ee45d3e9

Vendored third-party code (MIT). Its value is being diff-able against upstream,
so do not reformat, rename or tidy it. Deviations are marked ``# VENDOR:``.
"""

# ruff: noqa
# fmt: off
# (vendored: linting and formatting are off so the file stays byte-comparable
#  with upstream; do not remove these two directives)


# VENDOR: `PretrainedConfig` -> plain object; `transformers` is unimportable
# under torch 2.6 in this env and is not a train-time dependency.
class BatConfig:
    model_type = "bat"

    def __init__(
        self,
        input_shape=(1024, 128),
        patch_size=(16, 16),
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.0,
        layer_norm_eps=1e-6,
        pre_norm=True,
        layer_norm_first=False,
        init_scale_values=None,
        use_gate=True,
        pos_trainable=False,
        chunk_time_patches=64,
        sample_rate=16000,
        n_fft=1024,
        hop_length=160,
        n_mels=128,
        f_min=0,
        top_db=80,
        feature_normalization="per_sample_minmax_after_db",
        grad_checkpoint=False,
        **kwargs,
    ):
        # VENDOR: replaces `super().__init__(**kwargs)`. `BatModel.forward` reads
        # these three, so keep them with transformers' defaults; remaining kwargs
        # (`architectures`, `auto_map`, `pretraining_dataset`, ...) are carried
        # through as plain attributes the way `PretrainedConfig` did.
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.use_return_dict = kwargs.pop("return_dict", True)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.input_shape = list(input_shape)
        self.patch_size = list(patch_size)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.layer_norm_eps = layer_norm_eps
        self.pre_norm = pre_norm
        self.layer_norm_first = layer_norm_first
        self.init_scale_values = init_scale_values
        self.use_gate = use_gate
        self.pos_trainable = pos_trainable
        self.chunk_time_patches = chunk_time_patches
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.top_db = top_db
        self.feature_normalization = feature_normalization
        # VENDOR: BAT ships no gradient checkpointing; 513 tokens is ~2x our
        # other arms, so the block loop is checkpointable when this is set.
        self.grad_checkpoint = grad_checkpoint

    @property
    def num_patches(self):
        return (self.input_shape[0] // self.patch_size[0]) * (self.input_shape[1] // self.patch_size[1])
