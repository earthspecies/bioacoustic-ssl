"""The published HuggingFace model code for the Soundscape-MAE release.

Three modules are copied verbatim into the HuggingFace model repo, and import
nothing from this package so they work with no more than ``torch``,
``torchaudio`` and ``transformers`` installed:

* ``configuration_soundscape_mae`` — the config,
* ``modeling_soundscape_mae`` — the encoder and the classification head,
* ``feature_extraction_soundscape_mae`` — the mel front-end.

``conversion`` stays behind: it maps training checkpoints onto the published
state dicts and is only ever run by ``scripts/export_hf_model.py``.

Importing this package registers the classes with the ``Auto*`` factories, so a
locally-exported artifact loads without ``trust_remote_code``::

    import soundscape_ssl.hf  # noqa: F401
    model = AutoModel.from_pretrained("artifacts/base")

Loading straight from the Hub needs no import — the published ``config.json``
carries an ``auto_map`` pointing at the copied modules — but does need
``trust_remote_code=True``.

Note that the project env pins ``transformers>=5.6.1``, which needs torch >= 2.7
against this env's 2.6, so importing this package fails here. Run anything that
touches it under the override the repo uses for its other transformers-version
conflicts: ``uv run --with "transformers==4.57.1" ...``.
"""

from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModel,
    AutoModelForAudioClassification,
)

from .configuration_soundscape_mae import SoundscapeMAEConfig
from .feature_extraction_soundscape_mae import SoundscapeMAEFeatureExtractor
from .modeling_soundscape_mae import (
    SoundscapeMAEForAudioClassification,
    SoundscapeMAEModel,
    SoundscapeMAEPreTrainedModel,
)


def register_auto_classes() -> None:
    """Register the released classes with the ``Auto*`` factories.

    Idempotent: re-registering the same classes is a no-op, which matters because
    a process that has already loaded the artifact's remote code has registered
    them once already.
    """
    try:
        AutoConfig.register(SoundscapeMAEConfig.model_type, SoundscapeMAEConfig)
        AutoModel.register(SoundscapeMAEConfig, SoundscapeMAEModel)
        AutoModelForAudioClassification.register(
            SoundscapeMAEConfig, SoundscapeMAEForAudioClassification
        )
        AutoFeatureExtractor.register(SoundscapeMAEConfig, SoundscapeMAEFeatureExtractor)
    except ValueError:
        pass


register_auto_classes()

__all__ = [
    "SoundscapeMAEConfig",
    "SoundscapeMAEFeatureExtractor",
    "SoundscapeMAEForAudioClassification",
    "SoundscapeMAEModel",
    "SoundscapeMAEPreTrainedModel",
    "register_auto_classes",
]
