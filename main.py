import asyncio
import warnings

# Suppress noisy deprecation warnings from third-party libraries at startup.
# These are upstream issues and not actionable from our side.
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", message="dropout option adds dropout")
warnings.filterwarnings("ignore", message="`torch.nn.utils.weight_norm` is deprecated")

from iara.bot import main

if __name__ == "__main__":
    asyncio.run(main())
