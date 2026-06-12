"""Initialize database to default demo state.

Usage:
    python init_db.py          # create tables + seed if empty
    python init_db.py --reset  # wipe and restore default demo data
"""

import asyncio

from seed import init_database

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Initialize AV platform database")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop all data and restore default demo seed",
    )
    args = parser.parse_args()
    asyncio.run(init_database(reset=args.reset))
