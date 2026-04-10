import asyncio
import warnings
from core.app import main

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    asyncio.run(main())
