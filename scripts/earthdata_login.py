"""One-time NASA Earthdata login: persist credentials to ``~/.netrc``.

Reads ``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD`` (or ``EARTHDATA_TOKEN``)
from the environment and calls ``earthaccess.login(strategy="all",
persist=True)``, which writes ``~/.netrc`` (mode 600) so future runs on this
machine authenticate without env vars.

Run once::

    EARTHDATA_USERNAME=... EARTHDATA_PASSWORD=... uv run python scripts/earthdata_login.py
"""

from dotenv import load_dotenv
load_dotenv()  # load repo .env (secrets, HF cache, CA bundle) before other imports

from pathlib import Path

import earthaccess


def main() -> None:
    auth = earthaccess.login(strategy="all", persist=True)
    netrc = Path.home() / ".netrc"
    if auth.authenticated:
        print(f"Earthdata login OK (user: {auth.username}).")
        print(f"Credentials persisted to: {netrc} (exists={netrc.exists()})")
        # Print the EDL token so it can be added to .env as EARTHDATA_TOKEN,
        # which lets DataLoader workers authenticate fully offline (no per-worker
        # network token request, avoids the 503 login herd at high num_workers).
        token = auth.token.get("access_token") if auth.token else None
        if token:
            print(f"\nEARTHDATA_TOKEN={token}")
            print("(add the line above to .env for offline per-worker login)")
    else:
        raise SystemExit(
            "Earthdata login failed. Set EARTHDATA_USERNAME / EARTHDATA_PASSWORD "
            "(or EARTHDATA_TOKEN) and try again."
        )


if __name__ == "__main__":
    main()
