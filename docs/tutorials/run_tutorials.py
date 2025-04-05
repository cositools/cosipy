#!/usr/bin/env python3

import logging
import traceback

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

import argparse
import os
import shutil
import timeit
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter
from nbconvert.writers import FilesWriter

import numpy as np
import yaml

from yayc import Configurator

from cosipy.util import fetch_wasabi_file, fetch_wasabi_file_header


def main():

    p = argparse.ArgumentParser(description="Run one or all tutorials")
    p.add_argument("config", help="YAML config file")
    p.add_argument("--config_group", default='tutorials',
                   help="Path withing the config file with the tutorials information")
    p.add_argument("-o","--output", nargs='?',
                   help="Output directory. It will create subdirectories for each tutorials. "
                        "If not empty, all files will be removed.")
    p.add_argument("--override", nargs='*',
                   help="Override config parameters. e.g. \"section:param_int = 2\" \"section:param_string = b\"")
    p.add_argument("--wasabi_mirror", nargs='?',
                   help=("Path to local wasabi mirror. We will try to symlink existing file from there."
                         "Otherwise they will be downloaded. If provided and the needed files do not exists, "
                         "they will be cached here. Overrides path in config."))
    p.add_argument('--tutorial', nargs='*', default = None,
                   help = "Which tutorials to run. All by default.")
    p.add_argument('--log-level', default='info',
                    help='Set the logging level (debug, info, warning, error, critical)')
    p.add_argument('--dry', action='store_true', default=False,
                   help='Perform the setup, fetching and checks, but do not execute the notebooks')
    args = p.parse_args()

    # Logger
    logger.setLevel(level=args.log_level.upper())

    # config file
    full_config = Configurator.open(args.config)
    config = Configurator(full_config[args.config_group])
    config.config_path = full_config.config_path
    if args.override is not None:
        config.override(*args.override)

    wasabi_bucket = config['globals:wasabi_bucket']
    wasabi_mirror = args.wasabi_mirror

    if wasabi_mirror is None:
        wasabi_mirror = config['globals:wasabi_mirror']

    if wasabi_mirror is not None:
        wasabi_mirror = Path(wasabi_mirror)

    output_dir = args.output
    if output_dir is None:
        output_dir = config['globals:output_dir']

    if output_dir is None:
        raise p.error("Must provide output directory, either in config file or command line.")
    else:
        output_dir = Path(output_dir)

    logger.info(f"Config:\n{config.dump()}")

    # Which tutorials to run
    tutorials = args.tutorial

    if tutorials is None:
        tutorials = list(config['tutorials'].keys())

    # Cache files
    def cache_wasabi_files(tutorial):
        """
        Cache all file for a given tutorial
        """
        if 'wasabi_files' in config['tutorials'][tutorial]:
            for file in config['tutorials'][tutorial]['wasabi_files']:
                output = wasabi_mirror/file
                logger.info(f"Fetching {file} to {output}")
                output.parent.mkdir(parents=True, exist_ok=True)
                metadata = fetch_wasabi_file(file, output, overwrite=True, bucket=wasabi_bucket)
                logger.info(yaml.dump(metadata))

        if 'wasabi_files_unzip' in config['tutorials'][tutorial]:
            for file,file_args in config['tutorials'][tutorial]['wasabi_files_unzip'].items():
                output = wasabi_mirror / file

                unzip_output = None
                if 'unzip_output' in file_args:
                    unzip_output = output.parent/file_args['unzip_output']

                checksum = None
                if 'checksum' in file_args:
                    checksum = file_args['checksum']

                logger.info(f"Fetching {file} and unzipping it to {unzip_output}")
                output.parent.mkdir(parents=True, exist_ok=True)
                metadata = fetch_wasabi_file(file, output, overwrite=True, bucket=wasabi_bucket, unzip = True, unzip_output=unzip_output, checksum=checksum)
                logger.info(yaml.dump(metadata))

    if wasabi_mirror is not None:
        for tutorial in tutorials:
            cache_wasabi_files(tutorial)

    # Run the tutorials
    def run_tutorial(tutorial):
        """
        Run specific tutorial
        """
        # Clear working directory
        wdir = output_dir/tutorial
        if wdir.exists():
            shutil.rmtree(wdir)
        wdir.mkdir(parents=True)

        # Add logger to file
        file_handler = logging.FileHandler(wdir/"run.log", mode='a')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        # Copy notebook and ancillary files
        notebooks = np.atleast_1d(config['tutorials'][tutorial]['notebook'])

        for notebook in notebooks:
            source_nb = config.absolute_path(notebook)
            shutil.copyfile(source_nb, wdir/source_nb.name)

        if 'ancillary_files' in config['tutorials'][tutorial]:
            for file in config['tutorials'][tutorial]['ancillary_files']:
                source_file = config.absolute_path(file)
                shutil.copyfile(source_file, wdir/source_file.name)

        # Link wasabi files from cache is they exists
        if wasabi_mirror is not None:
            if 'wasabi_files' in config['tutorials'][tutorial]:
                for rel_path in config['tutorials'][tutorial]['wasabi_files']:
                    local_copy = wasabi_mirror/rel_path

                    if local_copy.exists():
                        os.symlink(local_copy, wdir/local_copy.name)

            if 'wasabi_files_unzip' in config['tutorials'][tutorial]:
                for rel_path,other_args in config['tutorials'][tutorial]['wasabi_files_unzip'].items():

                    if 'output_unzip' in other_args:
                        # The unzipped file as a different name than just dropping .gz or .zip
                        local_copy = wasabi_mirror / other_args['output_unzip']
                    else:
                        local_copy = wasabi_mirror / rel_path

                    if local_copy.exists():
                        os.symlink(local_copy, wdir / local_copy.name)

        # Run
        if not args.dry:
            for notebook in notebooks:
                source_nb_path = config.absolute_path(notebook)
                nb_path = wdir/source_nb_path.name

                with (open(nb_path) as nb_file):
                    nb = nbformat.read(nb_file, as_version=nbformat.NO_CONVERT)

                    logger.info(f"Executing notebook {source_nb_path}...")
                    start_time = timeit.default_timer()
                    ep = ExecutePreprocessor(timeout=config['globals:timeout'], kernel_name=config['globals:kernel'])
                    ep_out = ep.preprocess(nb, {'metadata': {'path': str(wdir)}})
                    elapsed = timeit.default_timer() - start_time
                    logger.info(f"Notebook {source_nb_path} took {elapsed} seconds to finish.")

                    nb_exec_path = nb_path.with_name(nb_path.stem + "_executed" + nb_path.suffix)
                    with open(nb_exec_path, 'w', encoding='utf-8') as exec_nb_file:
                        nbformat.write(nb, exec_nb_file)
                        logger.info(f"Saved executed file to {nb_exec_path}")

                    html_path = nb_exec_path.with_suffix('.html')
                    html_exporter = HTMLExporter(template_name="classic")
                    (body, resources) = html_exporter.from_notebook_node(nb)
                    html_writer = FilesWriter()
                    html_writer.write(body, resources, notebook_name = str(html_path.with_suffix('')))

        # Remove file logger
        logger.removeHandler(file_handler)

    summary = {}
    for tutorial in tutorials:

        summary[tutorial] = {}
        summary_entry = summary[tutorial]

        start_time = timeit.default_timer()
        try:
            run_tutorial(tutorial)
        except Exception as e:
            logger.error(f"Tutorial {tutorial} failed. Error:\n{e}")
            traceback.print_exc()
            succeeded = False
        else:
            succeeded = True

        elapsed = timeit.default_timer() - start_time

        summary_entry['succeeded'] = succeeded
        summary_entry['elapsed_sec'] = elapsed

    logger.info(f"Run summary:\n{yaml.dump(summary)}")


if __name__ == "__main__":
    main()

