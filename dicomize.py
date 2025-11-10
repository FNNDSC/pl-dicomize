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
import hashlib
import uuid

__version__ = '1.0.4'

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
parser.add_argument(
    '--conceptName',
    type=str,
    default="",
    help="Specify the header of concept sequence. Required for SR generation"
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


def anonymize_uid_deterministic(seed: str, root: str = "2.25.") -> str:
    """Generate a deterministic, valid DICOM UID from an input string."""
    u = uuid.uuid5(uuid.NAMESPACE_URL, seed)
    # Convert UUID to digits and remove leading zeros
    digits = str(u.int).lstrip("0")
    # Concatenate and enforce 64-char limit
    uid = (root + digits)[:64].rstrip(".")
    return uid

def apply_concept_info(ds, concept_name):
    concept_item = Dataset()
    ds.ValueType = "CONTAINER"
    concept_item.CodeValue = concept_name
    concept_item.CodingSchemeDesignator = format_string(concept_name)
    concept_item.CodeMeaning = concept_name

    ds.ConceptNameCodeSequence = Sequence([concept_item])
    ds.ContinuityOfContent = "SEPARATE"
    ds.CompletionFlag = "COMPLETE"
    ds.VerificationFlag = "VERIFIED"
    ds.PerformedProcedureCodeSequence = Sequence([])

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
        ds.Manufacturer = "ChRIS"


def create_base_dataset():
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # Default SR SOP class uid
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.88.11"
    ds.SOPInstanceUID = generate_uid()
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "SR"
    ds.ContentDate = datetime.now().strftime('%Y%m%d')
    ds.ContentTime = datetime.now().strftime('%H%M%S')
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.Manufacturer = "ChRIS"
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
    ds.save_as(output_path)
    print(f"Saved DICOM to: ----> {output_path} <----\n")

def create_dicom(
        json_data: dict,
        output_path: Path=None,
        dicom_path: Path=None,
        image_path: Path=None,
        image_type: str=None,
        tags_to_copy: str=None,
        concept_name: str=None
):
    """Main DICOM creation logic."""
    ds = create_base_dataset()

    # Create dicom from an image
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

    # Create dicom from existing dicom
    elif dicom_path:
        ds = read_dicom(dicom_path)

        fields = [
            "PixelData",
            "Rows",
            "Columns",
            "PhotometricInterpretation",
            "FieldOfViewDimensions",
            "SamplesPerPixel",
            "BitsAllocated",
            "BitsStored",
            "HighBit",
            "PixelRepresentation",
            "SeriesNumber",
            "SOPClassUID",
            "InstanceNumber",
            "Modality",
            "NumberOfFrames",
            "StudyInstanceUID",
            "SeriesInstanceUID"
        ]

        # copy additional tags specified to preserve from CLI
        if tags_to_copy:
            for tag in tags_to_copy:
                fields.append(tag)

        all_tags = [elem.keyword for elem in ds if elem.keyword]

        for tag in all_tags:
            if tag not in fields:
                delattr(ds, tag)
            else:
                print(f"Preserved tag: {tag}")

        ds.remove_private_tags()

        ds.StudyInstanceUID = anonymize_uid_deterministic(ds.StudyInstanceUID)
        ds.SeriesInstanceUID = anonymize_uid_deterministic(ds.SeriesInstanceUID)
        ds.SOPInstanceUID = generate_uid()

    # Create Structure Report from JSON
    else:
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        output_path = os.path.join(output_path,f"0001-{ds.SOPInstanceUID}.dcm")

    if concept_name:
        apply_concept_info(ds, concept_name)

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
    min_memory_limit='1000Mi',  # supported units: Mi, Gi
    min_cpu_limit='1000m',  # millicores, e.g. "1000m" = 1 CPU core
    min_gpu_limit=0  # set min_gpu_limit=1 to enable GPU
)
def main(options: Namespace, inputdir: Path, outputdir: Path):

    print(DISPLAY_TITLE)

    # tags to copy from existing DICOM
    tags = options.copy_tags.split(",") if options.copy_tags else []

    # Serialize JSON data from CLI args or json file
    json_data = serialize_json(options, inputdir)

    # handles multiple use cases of the plugin
    # 1) Create empty DICOM (Structured Reports)
    # 2) Create new DICOM from existing DICOM
    # 3) Create new DICOM from existing image
    match options.createFrom:
        case "empty":
            create_dicom(json_data, outputdir, concept_name=options.conceptName)
        case "dicom" | "image":
            mapper = PathMapper.file_mapper(inputdir, outputdir, glob=f"**/*{options.pattern}", fail_if_empty=True)
            for src, dst in mapper:
                create_dicom(
                    json_data,
                    output_path=dst,
                    dicom_path=src if options.createFrom == "dicom" else None,
                    image_path=src if options.createFrom == "image" else None,
                    image_type=options.pattern if options.createFrom == "image" else None,
                    tags_to_copy=tags,
                    concept_name=options.conceptName,
                )
        case _:
            print(f"Unknown --createFrom mode: {options.createFrom}")



if __name__ == '__main__':
    main()
