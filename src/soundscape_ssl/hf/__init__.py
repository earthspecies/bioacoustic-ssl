"""The published HuggingFace model code for the BirdMAE2 release.

Three modules are copied verbatim into the HuggingFace model repo, and import
nothing from this package so they work with no more than ``torch``,
``torchaudio`` and ``transformers`` installed:

* ``configuration_birdmae2`` — the config,
* ``modeling_birdmae2`` — the encoder and the classification head,
* ``feature_extraction_birdmae2`` — the mel front-end.

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

from .configuration_birdmae2 import BirdMAE2Config
from .feature_extraction_birdmae2 import BirdMAE2FeatureExtractor
from .modeling_birdmae2 import (
    BirdMAE2ForAudioClassification,
    BirdMAE2Model,
    BirdMAE2PreTrainedModel,
)


def register_auto_classes() -> None:
    """Register the released classes with the ``Auto*`` factories.

    Idempotent: re-registering the same classes is a no-op, which matters because
    a process that has already loaded the artifact's remote code has registered
    them once already.
    """
    try:
        AutoConfig.register(BirdMAE2Config.model_type, BirdMAE2Config)
        AutoModel.register(BirdMAE2Config, BirdMAE2Model)
        AutoModelForAudioClassification.register(
            BirdMAE2Config, BirdMAE2ForAudioClassification
        )
        AutoFeatureExtractor.register(BirdMAE2Config, BirdMAE2FeatureExtractor)
    except ValueError:
        pass


register_auto_classes()

__all__ = [
    "BirdMAE2Config",
    "BirdMAE2FeatureExtractor",
    "BirdMAE2ForAudioClassification",
    "BirdMAE2Model",
    "BirdMAE2PreTrainedModel",
    "register_auto_classes",
]
