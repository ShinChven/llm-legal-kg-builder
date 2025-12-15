import os
import re
import textwrap
from dotenv import load_dotenv
import langextract as lx
import json
from typing import Dict, List

from src.re.act_relationship_handler import ActRelationshipHandler
from src.re.normalization import normalize_title

# Load environment variables from .env file
load_dotenv()


def extract_act_year(title: str) -> int | None:
    """Extracts the trailing year (4 digits) from an Act title.

    Accepts years 1800-2099 at the very end of the string (optionally followed by whitespace).
    Returns the int year or None if not found.
    """
    if not title:
        return None
    m = re.search(r'(18|19|20)\d{2}$', title.strip())
    if m:
        try:
            return int(m.group(0))
        except ValueError:
            return None
    return None


class LangExtractRelationshipExtractor:
    """
    Extracts legislative relationships from text using the langextract library
    and stores them in the database.
    """

    def __init__(self, use_cache: bool = True, model_id: str = "gemini-2.5-flash", max_workers: int = 10):
        """
        Initializes the extractor with a model, database handler, and configuration.

        Args:
            use_cache: If True, uses existing database relationships instead of re-extracting.
            model_id: The ID of the Gemini model to use.
            max_workers: The number of parallel workers for extraction.
        """
        self.act_handler = ActRelationshipHandler()
        self.use_cache = use_cache
        self.model_id = model_id
        self.max_workers = max_workers
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file or environment variables.")

        # 1. Define the prompt and extraction rules
        self.prompt = textwrap.dedent("""
            You are an expert legal analyst specializing in New Zealand legislation.
            Your task is to extract relationships between legislative documents.

            CRITICAL: Only extract relationships where the target is a complete Act title with a year
            (e.g., \"Privacy Act 2020\", \"Social Security Act 2018\").
            Ignore regulations, rules, orders, bills, or other non-Act legislation.

            The relationship types are:
            - AMD (Amendment): The source document amends the target Act.
            - PRP (Partial Repeal): The source document partially repeals the target Act.
            - FRP (Full Repeal): The source document completely repeals the target Act.
            - CIT (Citation): The source document cites or refers to the target Act for other reasons.

            Use the exact text for the extracted Act title. Do not paraphrase.
            The 'relationship_type' attribute must be one of the four codes above.
            """)

        # 2. Provide a high-quality example to guide the model
        self.examples = [
            lx.data.ExampleData(
                text="This Act may be cited as the Fictional Amendment Act 2024. Section 5 amends the Principal Act, "
                     "the Imaginary Property Act 2022, by inserting a new section 10A. Furthermore, the Obsolete "
                     "Regulations Act 1999 is hereby repealed. This Act also makes reference to the Contracts and "
                     "Commercial Law Act 2017 for definitions.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="act_relationship",
                        extraction_text="Imaginary Property Act 2022",
                        attributes={"relationship_type": "AMD"}
                    ),
                    lx.data.Extraction(
                        extraction_class="act_relationship",
                        extraction_text="Obsolete Regulations Act 1999",
                        attributes={"relationship_type": "FRP"}
                    ),
                    lx.data.Extraction(
                        extraction_class="act_relationship",
                        extraction_text="Contracts and Commercial Law Act 2017",
                        attributes={"relationship_type": "CIT"}
                    ),
                ]
            )
        ]

    def extract_and_store_relationships(self, title: str, visualize: bool = False, output_dir: str = "."):
        """
        Extracts relationships for a given act title, stores them in the database,
        and optionally creates a visualization.

        Args:
            title: The title of the source legislative act.
            visualize: If True, generates an interactive HTML visualization file.
            output_dir: The directory to save visualization files.
        """
        print(f"Processing '{title}'...")

        if self.use_cache:
            existing_relations = self.act_handler.get_relationships_by_subject(title)
            if existing_relations:
                print(f"Found {len(existing_relations)} existing relationships for '{title}'. Skipping extraction.")
                return

        try:
            full_text = self._retrieve_legislation(title)
            if not full_text:
                self.act_handler.mark_as_processed(title) # Mark as processed even if no text
                return

            # Perform the extraction
            target_documents = self._extract_with_langextract(full_text, visualize, output_dir)

            # Clean and store the results
            if target_documents:
                cleaned_targets = self._clean_and_merge_target_documents(target_documents)

                # Filter out relationships to self or where object Act year is newer than subject Act year
                subject_year = extract_act_year(title)
                if subject_year:
                    filtered_targets = {}
                    for object_title, rel_codes in cleaned_targets.items():
                        # Rule 1: An Act cannot have a relationship with itself.
                        if normalize_title(object_title) == normalize_title(title):
                            print(f"Discarding self-relationship for '{title}'.")
                            continue

                        # Rule 2: The object Act cannot be newer than the subject Act.
                        object_year = extract_act_year(object_title)
                        if object_year and object_year > subject_year:
                            print(f"Discarding relationship to '{object_title}' (year {object_year}) as it is newer than subject '{title}' ({subject_year}).")
                            continue

                        filtered_targets[object_title] = rel_codes
                    cleaned_targets = filtered_targets

                print(f"Storing {len(cleaned_targets)} cleaned relationships for '{title}' in database.")
                for object_name, relationships in cleaned_targets.items():
                    normalized_object_name = normalize_title(object_name)
                    self.act_handler.upsert_relationship(title, normalized_object_name, relationships)

            # Mark as processed after successful extraction and storage
            self.act_handler.mark_as_processed(title)
            print(f"Successfully processed and stored relationships for '{title}'.")

        except ValueError as e:
            print(f"Could not process '{title}': {e}")
        except Exception as e:
            print(f"An unexpected error occurred while processing '{title}': {e}")


    def _retrieve_legislation(self, title: str) -> str:
        """
        Fetches the full text of a legislative document from the database.
        """
        conn = self.act_handler.get_connection()
        if conn is None:
            raise ConnectionError("Failed to get a database connection.")
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT text FROM legislations WHERE title = %s LIMIT 1", (title,))
                result = cursor.fetchone()
                if not result or not result[0]:
                    raise ValueError(f"Legislation '{title}' not found or has no text.")
                return result[0]
        finally:
            self.act_handler.return_connection(conn)


    def _clean_and_merge_target_documents(self, target_documents: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Cleans and merges target documents by checking for 'The ' prefix variations against the database.
        """
        conn = self.act_handler.get_connection()
        if conn is None:
            print("Warning: DB connection not available for title cleaning. Proceeding without verification.")
            return target_documents

        try:
            title_groups: Dict[str, Dict[str, List[str]]] = {}
            for object_title, relationships in target_documents.items():
                normalized_title = normalize_title(object_title)
                base_title = normalized_title[4:].strip() if normalized_title.lower().startswith('the ') else normalized_title
                if base_title not in title_groups:
                    title_groups[base_title] = {}
                title_groups[base_title][normalized_title] = relationships

            cleaned_documents: Dict[str, List[str]] = {}
            with conn.cursor() as cursor:
                for base_title, variations in title_groups.items():
                    prefixed_title = f"The {base_title}"
                    cursor.execute("SELECT title FROM legislations WHERE LOWER(title) = LOWER(%s) LIMIT 1", (prefixed_title,))
                    db_result = cursor.fetchone()
                    canonical_title = db_result[0] if db_result else base_title

                    merged_relationships = set()
                    for rels in variations.values():
                        merged_relationships.update(rels)

                    if canonical_title in cleaned_documents:
                        cleaned_documents[canonical_title].update(merged_relationships)
                    else:
                        cleaned_documents[canonical_title] = merged_relationships

            return {title: sorted(list(rels)) for title, rels in cleaned_documents.items()}
        finally:
            self.act_handler.return_connection(conn)


    def _extract_with_langextract(self, text: str, visualize: bool = False, output_dir: str = "."):
        """
        Private method to perform the core relationship extraction using langextract.
        """
        print(f"Starting extraction with langextract on a text of {len(text)} characters...")

        result = lx.extract(
            text_or_documents=text,
            prompt_description=self.prompt,
            examples=self.examples,
            model_id=self.model_id,
            api_key=self.api_key,
            max_workers=self.max_workers,
            extraction_passes=2,
            max_char_buffer=2000,
        )

        target_documents = {}
        for extraction in result.extractions:
            if extraction.extraction_class == "act_relationship":
                title = extraction.extraction_text
                rel_type = extraction.attributes.get("relationship_type")
                if title and rel_type:
                    if title not in target_documents:
                        target_documents[title] = set()
                    target_documents[title].add(rel_type)

        final_results = {title: sorted(list(rels)) for title, rels in target_documents.items()}
        print(f"Extraction complete. Found {len(final_results)} related acts.")

        if visualize:
            output_path = os.path.join(output_dir, "langextract_visualization.jsonl")
            print(f"Saving extraction data to {output_path}...")
            lx.io.save_annotated_documents([result], output_name=os.path.basename(output_path), output_dir=output_dir)

            html_content = lx.visualize(output_path)
            viz_path = os.path.join(output_dir, "langextract_visualization.html")
            with open(viz_path, "w", encoding="utf-8") as f:
                f.write(getattr(html_content, 'data', str(html_content)))
            print(f"Visualization saved to {viz_path}")

        return final_results
