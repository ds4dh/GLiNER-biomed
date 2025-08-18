import os
import json
from lxml import etree
from tqdm.auto import tqdm

def extract_setid_drug_mapping(directory):
    """
    Extracts `setId`, `drug name`, and `synonyms` from all XML files in the given directory.

    Args:
        directory (str): Path to the directory containing HPL XML files.

    Returns:
        list, dict: A list of `setId` values and a dictionary mapping `setId` to a dictionary with `name` and `synonyms`.
    """
    set_ids = []
    setid_to_drug_mapping = {}
    parser = etree.XMLParser(huge_tree=True)  # Enable parsing of deeply nested XML

    # Loop through all files in the directory
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".xml"):
            file_path = os.path.join(directory, filename)
            try:
                # Parse the XML file with the configured parser
                tree = etree.parse(file_path, parser=parser)
                root = tree.getroot()

                # Extract the `setId` element
                setid_element = root.find(".//{urn:hl7-org:v3}setId")
                setid = setid_element.get("root") if setid_element is not None else None

                # Extract the drug name
                name_element = root.find(".//{urn:hl7-org:v3}name")
                drug_name = name_element.text.strip() if name_element is not None else None

                # Extract synonyms
                synonyms = []
                synonym_elements = root.findall(".//{urn:hl7-org:v3}genericMedicine/{urn:hl7-org:v3}name")
                if synonym_elements:
                    synonyms = list(set([syn.text.strip() for syn in synonym_elements if syn.text]))

                # Store data if setId is found
                if setid:
                    set_ids.append(setid)
                    setid_to_drug_mapping[setid] = {
                        "name": drug_name,
                        "synonyms": synonyms
                    }
                else:
                    print(f"No `setId` found in {filename}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

    return set_ids, setid_to_drug_mapping


if __name__ == "__main__":
    # Replace with the path to your directory containing the XML files
    xml_directory = "./data/hpl_xmls"
    txt_output_file = "./data/hpl_setids.txt"
    json_output_file = "./data/hpl_setid_drug_mapping.json"

    print(f"Processing XML files in: {xml_directory}")
    set_ids, setid_drug_mapping = extract_setid_drug_mapping(xml_directory)
    
    print(f"Retrieved {len(set_ids)} Set IDs.")

    # Save the set IDs to a text file
    with open(txt_output_file, "w") as f:
        for setid in set_ids:
            f.write(f"{setid}\n")
    print(f"Set IDs saved to {txt_output_file}")

    # Save the mapping to a JSON file
    with open(json_output_file, "w") as f:
        json.dump(setid_drug_mapping, f, indent=4)
    print(f"Set ID to drug name and synonyms mapping saved to {json_output_file}")