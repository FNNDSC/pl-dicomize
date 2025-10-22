#!/usr/bin/env python

from pathlib import Path
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
from chris_plugin import chris_plugin, PathMapper
import argparse
import os
import json
import pydicom
from pydicom.dataset import FileDataset, Dataset
from pydicom.tag import Tag
from pydicom.uid import (
    generate_uid,
    ExplicitVRLittleEndian,
    SecondaryCaptureImageStorage
)
from pydicom.datadict import keyword_dict, add_private_dict_entry
from datetime import datetime
import numpy as np
from PIL import Image
from pydicom.sequence import Sequence

__version__ = '1.0.0'

DISPLAY_TITLE = r"""
       _           _ _                     _         
      | |         | (_)                   (_)        
 _ __ | |______ __| |_  ___ ___  _ __ ___  _ _______ 
| '_ \| |______/ _` | |/ __/ _ \| '_ ` _ \| |_  / _ \
| |_) | |     | (_| | | (_| (_) | | | | | | |/ /  __/
| .__/|_|      \__,_|_|\___\___/|_| |_| |_|_/___\___|
| |                                                  
|_|                                                  
"""

parser = ArgumentParser(description='A DICOM generator ChRIS plugin',
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-V', '--version', action='version',
                    version=f'%(prog)s {__version__}')
parser.add_argument(
    "--pattern",
    default="dcm",
    help="""
            pattern for file names to include (you should quote this!)
            (this flag triggers the PathMapper on the inputdir).""",
)
parser.add_argument(
    '--jsonFile',
    type=str,
    default="json",
    help='Path to JSON file'
)
parser.add_argument(
    '--tagStruct',
    type=str,
    default="",
    help='DICOM headers as stringified JSON'
)
parser.add_argument(
    '--copy-tags',
    type=str,
    default="",
    help='Comma-separated list of tags to copy from existing DICOM'
)
parser.add_argument(
    '--createFrom',
    type=str,
    default="empty",
    help="Create new DICOM from existing: 1) dicom 2) image 3) empty"
)

def serialize_json(options, inputdir: Path):
    json_path_list = list(inputdir.glob(f"**/{options.jsonFile}"))
    json_path = json_path_list[0] if json_path_list else ""
    tag_dict = {}

    # Either json file or json structure could be specified but not both
    if options.tagStruct and json_path:
        print("Either json file or json structure could be specified but not both")
        return tag_dict
    if options.tagStruct:
        tag_dict = json.loads(options.tagStruct)
    if json_path:
        tag_dict = load_json(json_path)
    return tag_dict

def load_json(json_file):
    if not json_file:
        return {}
    with open(json_file, 'r') as f:
        print(f"Loading JSON file : ---->{json_file}<----")
        return json.load(f)


def load_image(image_path):
    print(f"Loading image file: ---->{image_path}<----")
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    # img = img.resize((512, 512))  # Resize to standard size
    arr = np.array(img).astype(np.uint8)
    return arr

def read_dicom(dicom_path):
    print(f"Reading input dicom file: ------>{dicom_path}<------")
    ds = pydicom.dcmread(dicom_path)
    return ds

def format_string(s):
    return s.upper().replace(" ", "_")

def apply_json_tags(ds, json_content):
    if not isinstance(json_content, dict):
        raise ValueError("JSON content must be a dictionary of tag names and values.")

    content_list = []

    for key, value in json_content.items():
        if key in keyword_dict:
            # Standard DICOM tag
            if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
                try:
                    value = eval(value)
                except Exception:
                    print(f"Failed to parse list for {key}: {value}")
            setattr(ds, key, value)
            print(f"Added standard tag: {key} = {value}")
        else:
            # Create ConceptNameCodeSequence item
            concept_item = Dataset()
            concept_item.CodeValue = key
            concept_item.CodingSchemeDesignator = format_string(key)
            concept_item.CodeMeaning = key
            concept_seq = Sequence([concept_item])

            # Create ContentSequence item
            content_item = Dataset()
            content_item.RelationshipType = "HAS PROPERTIES"
            content_item.ValueType = "TEXT"
            content_item.ConceptNameCodeSequence = concept_seq
            content_item.TextValue = value

            content_list.append(content_item)
            print(f"Added non-standard tag: {key} = {value}")

        # Create the full ContentSequence (with multiple items if needed)
        ds.ContentSequence = Sequence(content_list)


def create_base_dataset():
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPInstanceUID = generate_uid()
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "OT"
    ds.ContentDate = datetime.now().strftime('%Y%m%d')
    ds.ContentTime = datetime.now().strftime('%H%M%S')
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    return ds


def copy_selected_tags(source_ds, target_ds, tag_list):
    for tag in tag_list:
        if tag in keyword_dict:
            if hasattr(source_ds, tag):
                setattr(target_ds, tag, getattr(source_ds, tag))
                print(f"Copied tag: {tag}")
        else:
            print(f"Tag {tag} not found in DICOM dictionary.")

def add_dummy_pixel(ds):
    # Dummy pixel data
    IMAGE_SIZE = (128, 128)  # Rows, Columns
    # Image info
    rows, cols = IMAGE_SIZE
    ds.Rows = rows
    ds.Columns = cols
    pixel_array = (np.random.rand(rows, cols) * 65535).astype(np.uint16)
    ds.PixelData = pixel_array.tobytes()


def save_dataset(ds, output_path):
    ds.save_as(output_path, write_like_original=False)
    print(f"Saved DICOM to: ----> {output_path} <----\n")

def create_dicom(json_data, output_path=None, dicom_path=None, image_path=None, image_type=None, tags_to_copy=None):
    ds = create_base_dataset()

    # Use image pixel data
    if image_path:
        pixel_data = load_image(image_path)
        ds.Rows, ds.Columns = pixel_data.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.PixelData = pixel_data.tobytes()
        output_path = str(output_path).replace(image_type,"dcm")
    elif dicom_path:
        orig_ds = read_dicom(dicom_path)
        ds.PixelData = orig_ds.pixel_array.tobytes()
        ds.Rows = orig_ds.Rows
        ds.Columns = orig_ds.Columns
        ds.PhotometricInterpretation = orig_ds.PhotometricInterpretation
        ds.SamplesPerPixel = orig_ds.SamplesPerPixel
        ds.BitsAllocated = orig_ds.BitsAllocated
        ds.BitsStored = orig_ds.BitsStored
        ds.HighBit = orig_ds.HighBit
        ds.PixelRepresentation = orig_ds.PixelRepresentation
        if tags_to_copy:
            copy_selected_tags(orig_ds, ds, tags_to_copy)
    else:
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        output_path = os.path.join(output_path,f"0001-{ds.SOPInstanceUID}.dcm")

    # Apply JSON metadata
    apply_json_tags(ds, json_data)

    save_dataset(ds, output_path)


# The main function of this *ChRIS* plugin is denoted by this ``@chris_plugin`` "decorator."
# Some metadata about the plugin is specified here. There is more metadata specified in setup.py.
#
# documentation: https://fnndsc.github.io/chris_plugin/chris_plugin.html#chris_plugin
@chris_plugin(
    parser=parser,
    title='A DICOM generator plugin',
    category='',  # ref. https://chrisstore.co/plugins
    min_memory_limit='100Mi',  # supported units: Mi, Gi
    min_cpu_limit='1000m',  # millicores, e.g. "1000m" = 1 CPU core
    min_gpu_limit=0  # set min_gpu_limit=1 to enable GPU
)
def main(options: Namespace, inputdir: Path, outputdir: Path):
    """
    *ChRIS* plugins usually have two positional arguments: an **input directory** containing
    input files and an **output directory** where to write output files. Command-line arguments
    are passed to this main method implicitly when ``main()`` is called below without parameters.

    :param options: non-positional arguments parsed by the parser given to @chris_plugin
    :param inputdir: directory containing (read-only) input files
    :param outputdir: directory where to write output files
    """

    print(DISPLAY_TITLE)

    # tags to copy from existing DICOM
    tags = options.copy_tags.split(",") if options.copy_tags else []

    # Serialize JSON data from CLI args or json file
    json_dict = serialize_json(options, inputdir)

    # handles multiple use cases of the plugin
    # 1) Create empty DICOM
    # 2) Create new DICOM from existing DICOM
    # 3) Create new DICOM from existing image
    match options.createFrom:
        case "empty":
            create_dicom(
                json_data=json_dict,
                output_path=outputdir,
                dicom_path=None,
                image_path=None,
                tags_to_copy=None
            )
        case "dicom":
            # Create new DICOMs from existing ones
            mapper = PathMapper.file_mapper(inputdir, outputdir, glob=f"**/*{options.pattern}", fail_if_empty=False)
            for input_file, output_file in mapper:
                create_dicom(
                    json_data=json_dict,
                    output_path=output_file,
                    dicom_path=input_file,
                    image_path=None,
                    tags_to_copy=tags,
                )
        case "image":
            # Create new DICOMs from existing ones
            mapper = PathMapper.file_mapper(inputdir, outputdir, glob=f"**/*{options.pattern}", fail_if_empty=False)
            for input_file, output_file in mapper:
                create_dicom(
                    json_data=json_dict,
                    output_path=output_file,
                    dicom_path=None,
                    image_path=input_file,
                    image_type=options.pattern,
                    tags_to_copy=None,
                )
        case _:
            print(f"Unknown dicom creation mode specified: {options.createFrom}")



if __name__ == '__main__':
    main()
