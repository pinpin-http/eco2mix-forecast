"""Point d'entrée en ligne de commande : `python -m eco2mix_forecast <commande>`."""

from __future__ import annotations

import argparse
import logging
import os

# Limite les threads OpenMP/BLAS avant tout import numérique.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="eco2mix_forecast")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("data", help="télécharge les sources et construit les tables")
    sub.add_parser("backtest", help="backtest annuel, test final et comparaison à RTE")
    sub.add_parser("figures", help="génère les figures PNG dans docs/")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    if args.command == "data":
        from . import dataset, download
        download.download_all()
        dataset.build_all()
    elif args.command == "backtest":
        from . import backtest
        backtest.run_all()
    elif args.command == "figures":
        from . import report
        report.make_all()


if __name__ == "__main__":
    main()
