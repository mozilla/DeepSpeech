import argparse
import re

def get_importers_parser(description):
    parser = argparse.ArgumentParser(description=description)
    return parser

# Validate and normalize transcriptions. Returns a cleaned version of the label
# or None if it's invalid.
def validate_label_eng(label):
    # For now we can only handle [a-z ']
    if re.search(r"[0-9]|[(<\[\]&*{]", label) is not None:
        return None

    label = label.replace("-", " ")
    label = label.replace("_", " ")
    label = re.sub("[ ]{2,}", " ", label)
    label = label.replace(".", "")
    label = label.replace(",", "")
    label = label.replace(";", "")
    label = label.replace("?", "")
    label = label.replace("!", "")
    label = label.replace(":", "")
    label = label.replace("\"", "")
    label = label.strip()
    label = label.lower()

    return label if label else None
