import sys
from pathlib import Path
import logging
from datetime import datetime
from typing import *
import functools

import click

from fragment_feature_detection.converter import MzMLParser

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)

logger = logging.getLogger()


def arg_logger(f):
    @functools.wraps(f)
    def func(*args, **kwargs):
        logger.info("Start: %s -> args: %s, kwargs: %s" % (f.__name__, args, kwargs))
        res = f(*args, **kwargs)
        logger.info("Finish: %s" % f.__name__)
        return res

    return func


def configure_logger(fw: Optional[Callable] = None):
    def decorator(f: Callable):
        @functools.wraps(fw if fw else f)
        def func(*args, **kwargs):
            logger = logging.getLogger()
            context = click.get_current_context()
            subcommand = context.info_name
            path = context.params["path"]
            # Add file handler that logs into the experiment directory.
            file_handler = logging.FileHandler(
                "{path}/{:%Y%m%d-%H%M%S}__{cmd}.log".format(
                    datetime.now(), cmd=subcommand, path=path
                ),
                mode="w",
            )
            # Get default stream handler from root logger.
            stream_handler = [
                h
                for h in logging.getLogger().handlers
                if isinstance(h, logging.StreamHandler)
            ]
            if len(stream_handler) > 0:
                stream_handler = stream_handler[0]
                file_handler.setFormatter(stream_handler.formatter)
            logger.addHandler(file_handler)
            return f(*args, **kwargs)

        return func

    return decorator


@click.group()
def main():
    """ """
    pass


@arg_logger
def example() -> None:
    """
    Example func.

    Args:
        None
    Returns:
        None
    """
    pass


@main.command("example")
@click.option("--arg", help="")
@configure_logger(fw=example)
def _raw_to_mzml(*args: Any, **kwargs: Any) -> None:
    """ """
    example(*args, **kwargs)


@arg_logger
def convert_mzml_h5long(
    mzml_fh: Union[Path, str],
    out_loc: Optional[Path, str] = None,
) -> None:
    """ """
    mzml_fh = Path(mzml_fh)
    if not out_loc:
        out_loc = Path(mzml_fh).parent
    else:
        out_loc = Path(out_loc)

    logger.info(f'Beginning conversion sample {mzml_fh} to h5 long...')
    MzMLParser.to_ms2_h5(
        mzml_fh,
        h5_fh = out_loc / mzml_fh.with_suffix('.h5').name,   
    )
    
