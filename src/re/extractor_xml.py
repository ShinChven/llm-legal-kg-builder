import xml.etree.ElementTree as ET
import html
import sys
import json
import re
from src.db.db_connection import db_connection

def get_xml_by_title(title):
    """
    Retrieves XML content from the database for a given Act title.
    """
    conn = db_connection.get_connection()
    if conn:
        try:
            cursor = conn.cursor()
            # Use a LIKE query to find titles that contain the given string, case-insensitive
            cursor.execute("SELECT xml FROM legislations WHERE title ILIKE %s LIMIT 1", (f"%{title}%%",))
            result = cursor.fetchone()
            cursor.close()
            if result:
                return result[0]
            else:
                print(f"No legislation found with title containing: {title}")
                return None
        except Exception as e:
            print(f"Database query failed: {e}")
            return None
        finally:
            db_connection.release_connection(conn)
    else:
        print("Failed to get database connection.")
        return None

def extract_title(root):
    """
    Extracts and cleans the title of the act from the root of the XML tree.
    It first tries to get it directly from the <title> tag, and if that fails, it looks for specific patterns in the text content.
    """
    # First, try to extract directly from the <title> tag
    title_element = root.find(".//title")
    if title_element is not None and title_element.text:
        return html.unescape(title_element.text.strip())

    # If it cannot be found directly, search for specific introductory phrases
    citation_patterns = [
        "This Act may be cited as",
        "This Act is the",
        "This Act shall be known as",
        "This Act is called",
        "The Short Title of this Act is"
    ]

    for para in root.iter("para"):
        # Concatenate all text fragments in a paragraph
        full_text = "".join(para.itertext()).strip()
        for pattern in citation_patterns:
            if pattern in full_text:
                # Extract the text after the pattern as the title, and remove the period at the end
                act_title = full_text.split(pattern, 1)[-1].strip()
                return act_title.rstrip(".")
    return "Title Not Found"

def find_relation_codes(root):
    """
    Find all relation codes (amendments, repeals) in the XML tree.
    """
    relation_codes = set()

    # 1. Extract "AMD" and "PR" from history-note
    for history_note in root.findall(".//history-note"):
        operation = history_note.find(".//amending-operation")
        if operation is not None and operation.text:
            op_text = operation.text.strip().lower()
            if "repealed" in op_text or "omitted" in op_text:
                relation_codes.add("PR")
            elif "amended" in op_text or "inserted" in op_text or "substituted" in op_text:
                relation_codes.add("AMD")

    # 2. Extract "AMD_S", "R_S", "PR_S" from schedule
    for schedule in root.findall(".//schedule"):
        heading = schedule.find(".//heading")
        if heading is not None and heading.text:
            heading_text = heading.text.strip().lower()
            if "enactments amended" in heading_text or "consequential amendments" in heading_text:
                relation_codes.add("AMD_S")
            elif "enactments repealed" in heading_text:
                if "partially" in heading_text:
                     relation_codes.add("PR_S")
                else:
                     relation_codes.add("R_S")

    return sorted(list(relation_codes))

def extract_object_acts_with_relations(root, parent_map):
    """
    Extracts object acts and their relationship to the subject act.
    Returns a dictionary mapping object act titles to their relationship codes.

    The relationship types are:
    - AMD (Amendment) - when the source document amends another Act
    - PRP (Partial Repeal) - when the source document partially repeals another Act's section(s)
    - FRP (Full Repeal) - when the source document fully repeals another Act
    - CIT (Citation) - when the source document references another Act
    """
    object_acts = {}

    # 1. Extract acts from schedules with their relation codes
    for schedule in root.findall(".//schedule"):
        heading = schedule.find(".//heading")
        if heading is not None and heading.text:
            heading_text = heading.text.strip().lower()

            # Determine the relationship type from schedule heading
            relation_code = None
            if "enactments amended" in heading_text or "consequential amendments" in heading_text:
                relation_code = "AMD"
            elif "enactments repealed" in heading_text:
                if "partially" in heading_text or "section" in heading_text:
                    relation_code = "PRP"
                else:
                    relation_code = "FRP"

            # If we found a relation code, extract the acts from this schedule
            if relation_code:
                for element in schedule.findall(".//citation") + schedule.findall(".//leg-title"):
                    citation_text = "".join(element.itertext()).strip().replace('\u00a0', ' ')

                    # Check if the citation is likely a legislative act by looking for a year
                    if re.search(r'\b(19|20)\d{2}\b', citation_text):
                        # Clean the citation text
                        citation_text = citation_text.split(':')[0].strip()
                        if " to the " in citation_text:
                            citation_text = citation_text.split(" to the ")[-1].strip()
                        citation_text = re.sub(r'\s*\(\d{4}\s+No\s*\d+\)', '', citation_text).strip()

                        if citation_text not in object_acts:
                            object_acts[citation_text] = []
                        if relation_code not in object_acts[citation_text]:
                            object_acts[citation_text].append(relation_code)

    # 2. Extract amendments and repeals from history notes
    for history_note in root.findall(".//history-note"):
        # Look for cited acts in amendment contexts
        for element in history_note.findall(".//citation") + history_note.findall(".//leg-title"):
            citation_text = "".join(element.itertext()).strip().replace('\u00a0', ' ')

            if re.search(r'\b(19|20)\d{2}\b', citation_text):
                # Clean the citation text
                citation_text = citation_text.split(':')[0].strip()
                if " to the " in citation_text:
                    citation_text = citation_text.split(" to the ")[-1].strip()
                citation_text = re.sub(r'\s*\(\d{4}\s+No\s*\d+\)', '', citation_text).strip()

                # Determine relationship from the history note context
                history_text = "".join(history_note.itertext()).lower()
                relation_code = None

                if "repealed" in history_text:
                    if "section" in history_text or "partially" in history_text:
                        relation_code = "PRP"
                    else:
                        relation_code = "FRP"
                elif any(word in history_text for word in ["amended", "inserted", "substituted"]):
                    relation_code = "AMD"

                if relation_code:
                    if citation_text not in object_acts:
                        object_acts[citation_text] = []
                    if relation_code not in object_acts[citation_text]:
                        object_acts[citation_text].append(relation_code)

    # 3. Extract all citations and determine their relationship type
    for element in root.findall(".//citation") + root.findall(".//leg-title"):
        citation_text = "".join(element.itertext()).strip().replace('\u00a0', ' ')

        # Check if the citation is likely a legislative act by looking for a year
        if re.search(r'\b(19|20)\d{2}\b', citation_text):
            # Clean the citation text
            citation_text = citation_text.split(':')[0].strip()
            if " to the " in citation_text:
                citation_text = citation_text.split(" to the ")[-1].strip()
            citation_text = re.sub(r'\s*\(\d{4}\s+No\s*\d+\)', '', citation_text).strip()

            # Skip if we already processed this act from schedules or history notes
            if citation_text in object_acts:
                continue

            # Get the context to determine relationship type
            parent = parent_map.get(element)
            context_text = ""
            current = element

            # Look up the parent hierarchy to get broader context
            while current is not None and len(context_text) < 500:
                if current.text:
                    context_text = current.text + " " + context_text
                for child in current:
                    if child.text:
                        context_text += " " + child.text
                    if child.tail:
                        context_text += " " + child.tail
                current = parent_map.get(current)

            context_text = context_text.lower()

            # Determine relationship type based on context
            relation_code = None

            # Check for amendment patterns
            if any(word in context_text for word in ["amend", "amended", "amending", "insert", "inserted", "substitute", "substituted", "add", "added"]):
                relation_code = "AMD"
            # Check for repeal patterns
            elif any(word in context_text for word in ["repeal", "repealed", "repealing", "omit", "omitted"]):
                if any(word in context_text for word in ["section", "subsection", "paragraph", "part", "partially"]):
                    relation_code = "PRP"
                else:
                    relation_code = "FRP"
            # Default to citation if no other relationship is found
            else:
                relation_code = "CIT"

            # Add the act with its relationship
            if citation_text not in object_acts:
                object_acts[citation_text] = []
            if relation_code not in object_acts[citation_text]:
                object_acts[citation_text].append(relation_code)

    # Sort the relation codes for each act
    for act in object_acts:
        object_acts[act] = sorted(object_acts[act])

    return object_acts

def parse_legislation_xml(xml_string):
    """
    Parses the XML string, extracts the title, relation codes, and citations,
    and returns a dictionary.
    """
    if not xml_string:
        print("Error: No input received.")
        return None

    try:
        # Remove <end> tags before parsing
        xml_string = re.sub(r'<end[^>]*>.*?</end>', '', xml_string, flags=re.DOTALL | re.IGNORECASE)
        xml_string = re.sub(r'<end[^>]*/?>', '', xml_string, flags=re.IGNORECASE)

        root = ET.fromstring(xml_string)
        parent_map = {c: p for p in root.iter() for c in p}
        title = extract_title(root)
        object_acts = extract_object_acts_with_relations(root, parent_map)

        return object_acts

    except ET.ParseError as e:
        print(f"XML parsing error: {e}")
        return None

def process_act_by_title(title):
    """
    Processes an Act by its title, fetching XML from the database.
    """
    print(f"Attempting to process Act by title: {title}")
    xml_content = get_xml_by_title(title)
    if xml_content:
        result_dict = parse_legislation_xml(xml_content)
        if result_dict:
            print("\n" + "="*20)
            print("Successfully extracted information, the generated dictionary is as follows:")
            print(json.dumps(result_dict, indent=4))
            print("="*20)
    else:
        print(f"Could not process {title}, as no XML content was found.")


# --- Main program ---
if __name__ == '__main__':
    # Hardcode the title of the Act you want to process
    act_title_to_process = "Stamp and Cheque Duties Act 1971"
    process_act_by_title(act_title_to_process)
