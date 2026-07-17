import warnings

# gcsfs -> google.auth emits this when using gcloud end-user creds without a
# quota project; harmless for our anon/public-bucket access, so mute it.
warnings.filterwarnings(
    "ignore",
    message=".*end user credentials from Google Cloud SDK without a quota project.*",
)

from .data import utils
from .data.datasets import xeno_canto
