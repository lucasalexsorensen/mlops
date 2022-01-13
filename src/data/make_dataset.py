# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from PIL import Image
from glob import glob
from tqdm import tqdm
import os
import re

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    for s in ['Train', 'Test']:
        for cs in ['Mask', 'Non Mask']:
            c = '%s/%s' % (s, cs)
            input_images = glob('%s/%s/*.*' % (input_filepath, c))
            os.makedirs('%s/%s' % (output_filepath, c), exist_ok=True)
            for file in tqdm(input_images):
                im = Image.open(file)
                im.resize((64,64)).save('%s/%s/%s' % (output_filepath, c, os.path.basename(file).replace('jpg', 'png')))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
