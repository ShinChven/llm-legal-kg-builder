import json
import concurrent.futures
import time
from typing import Callable, Dict, List
from google import genai
from src.re.act_relationship_handler import ActRelationshipHandler


class ReassuranceExtractor:
    """
    Handles the reassurance of legislative relationships against full-text documents.
    """

    def __init__(self, model_name: str, reassurance_batch_size: int, max_workers: int, act_handler: ActRelationshipHandler):
        """
        Initialize the reassurance extractor.

        Args:
            model_name: The name of the generative AI model to use.
            reassurance_batch_size: The number of relationships to send in a single batch.
            max_workers: The maximum number of threads to use for parallel processing.
            act_handler: An instance of ActRelationshipHandler to interact with the database.
        """
        self.model_name = model_name
        self.reassurance_batch_size = reassurance_batch_size
        self.max_workers = max_workers
        self.act_handler = act_handler
        self.genai_client = genai.Client()
        self.last_not_found_acts: List[str] = []

    def _retrieve_legislation(self, title: str) -> str:
        """
        Fetches the full text of a legislative document from the database.
        """
        conn = self.act_handler.get_connection()
        if conn is None:
            raise ConnectionError("Failed to get a database connection from the pool.")
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT text FROM legislations WHERE title = %s ORDER BY updated_at DESC LIMIT 1",
                    (title,)
                )
                result = cursor.fetchone()
                if not result or not result[0]:
                    raise ValueError(
                        f"Legislation with title '{title}' not found or has no text."
                    )
                return result[0]
        finally:
            if conn:
                self.act_handler.return_connection(conn)

    def reassure_relationships(self, source_act_title: str, target_relationships: Dict[str, List[str]], enable_logging: bool = False) -> Dict[str, List[str]]:
        """
        Reassures a list of target relationships against the full text of a source act.

        Args:
            source_act_title: The title of the source act.
            target_relationships: A dictionary of target acts and their supposed relationships.
            enable_logging: Flag to enable print statements for progress.

        Returns:
            A dictionary of reassured and verified relationships.
        """
        if not target_relationships:
            if enable_logging:
                print("No target relationships to reassure. Returning empty dictionary.")
            return {}

        if enable_logging:
            print(f"--- Starting Reassurance Process for '{source_act_title}' ---")
            print(f"Found {len(target_relationships)} target acts to reassure.")

        try:
            full_text = self._retrieve_legislation(source_act_title)
        except (ValueError, ConnectionError) as e:
            if enable_logging:
                print(f"Error retrieving source act text: {e}. Aborting reassurance.")
            return {}  # Return empty if source act can't be loaded

        reassurance_llm_function = self._create_reassurance_llm_function(source_act_title, full_text)

        reassured_relationships = {}
        acts_not_found = set()
        target_titles = list(target_relationships.keys())
        batches = [target_titles[i:i + self.reassurance_batch_size] for i in range(0, len(target_titles), self.reassurance_batch_size)]

        def process_batch(batch: List[str], batch_num: int):
            """Processes a single batch of relationships."""
            if enable_logging:
                print(f"Processing batch {batch_num} with {len(batch)} acts...")

            max_retries = 20
            delay_seconds = 20
            for attempt in range(max_retries):
                try:
                    if enable_logging:
                        print(f"Attempt {attempt + 1}/{max_retries} for batch {batch_num}...")
                    result = reassurance_llm_function(batch)
                    # Treat missing/invalid key as a transient failure to enable retry
                    if not isinstance(result, dict) or "reassured_documents" not in result:
                        raise ValueError("Malformed reassurance response: missing 'reassured_documents'")

                    reassured = result.get("reassured_documents") or {}
                    if not isinstance(reassured, dict):
                        raise ValueError("Malformed reassurance response: 'reassured_documents' is not an object")

                    reported_missing = result.get("missing_target_acts") or []
                    if not isinstance(reported_missing, list):
                        raise ValueError("Malformed reassurance response: 'missing_target_acts' is not a list")

                    batch_not_found = set()
                    valid_results = {}

                    # Evaluate each requested title to track confirmed and missing acts
                    for title in batch:
                        codes = reassured.get(title)
                        if codes:
                            valid_results[title] = codes
                        else:
                            batch_not_found.add(title)

                    # Merge explicit LLM misses (filtered to the current batch to avoid cross-contamination)
                    for title in reported_missing:
                        if title in batch:
                            batch_not_found.add(title)

                    # Preserve any extra acts returned by the LLM that were not in the batch
                    for title, codes in reassured.items():
                        if title not in valid_results and title not in batch_not_found and codes:
                            valid_results[title] = codes

                    if enable_logging:
                        print(
                            f"Batch {batch_num} reassured successfully. "
                            f"Found {len(valid_results)} confirmed relationships. "
                            f"Flagged {len(batch_not_found)} acts as not found."
                        )

                    return {
                        "found": valid_results,
                        "not_found": sorted(batch_not_found)
                    }
                except Exception as e:
                    if enable_logging:
                        print(f"Error reassuring batch {batch_num} on attempt {attempt + 1}/{max_retries}: {e}")
                    if attempt < max_retries - 1:
                        if enable_logging:
                            print(f"Retrying in {delay_seconds}s...")
                        time.sleep(delay_seconds)
                    else:
                        if enable_logging:
                            print(f"Failed to reassure batch {batch_num} after {max_retries} attempts.")
                        return {}
            return {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {executor.submit(process_batch, batch, i+1): batch for i, batch in enumerate(batches)}
            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    data = future.result()
                    if not data:
                        continue

                    confirmed = data.get("found", {}) if isinstance(data, dict) else {}
                    missing = data.get("not_found", []) if isinstance(data, dict) else []

                    if confirmed:
                        reassured_relationships.update(confirmed)
                        for object_name, codes in confirmed.items():
                            self.act_handler.update_relationship_codes(source_act_title, object_name, codes)

                    if missing:
                        for object_name in missing:
                            if object_name not in acts_not_found:
                                deleted = self.act_handler.delete_relationship(source_act_title, object_name)
                                if enable_logging:
                                    if deleted:
                                        print(f"Removed relationship for not-found act: {source_act_title} -> {object_name}")
                                    else:
                                        print(f"No relationship entry to remove for not-found act: {source_act_title} -> {object_name}")
                            acts_not_found.add(object_name)
                except Exception as exc:
                    if enable_logging:
                        print(f'A batch generated an exception: {exc}')

        if enable_logging:
            print(f"\n--- Reassurance Process Completed for '{source_act_title}' ---")
            print(f"Total reassured relationships found: {len(reassured_relationships)}")
            if acts_not_found:
                print("Acts flagged as not found (relationships removed from database):")
                for object_name in sorted(acts_not_found):
                    print(f"  - {object_name}")

        # Store non-found acts for potential downstream access
        self.last_not_found_acts = sorted(acts_not_found)

        return reassured_relationships

    def _create_reassurance_llm_function(self, legislation_title: str, full_text: str) -> Callable[[List[str]], Dict]:
        """
        Creates the LLM function for the reassurance process.
        """
        def reassurance_llm_function(target_acts_batch: List[str]) -> Dict:
            target_acts_list_str = "\n".join([f"- {title}" for title in target_acts_batch])

            prompt = f"""
            You are an expert legal analyst specializing in New Zealand Legislation. Your task is to verify (reassure) a pre-existing list of legislative relationships.

            You will be given the full text of a source legislative act and a list of target acts that are believed to be related to it. You must carefully read the full text and confirm whether each target act is indeed mentioned and what the relationship is.

            **Source Act Title:** "{legislation_title}"

            **Full Text of Source Act:**
            ```
            {full_text}
            ```

            **Proposed Target Acts to Verify:**
            ```
            {target_acts_list_str}
            ```

            **Your Task:**
            1.  You must search the full text for mentions of each target act.
            2.  Evaluate the context of each mention to decide its relationship type.

            **Relationship Types:**
			- AMD (Amendment) - when the source document amends another Act
    		- PRP (Partial Repeal) - when the source document partially repeals one or more parts of the target act, look for if the document repeals a section, a subsection, a schedule or any part of the target act.
    		- FRP (Full Repeal) - when the source document completely or fully repeals the target act, which means there is *NO EVIDENCE* of the given document repeals a section, a subsection, a schedule or any part of the target act.
    		- CIT (Citation) - when you can't decide the relationship or it doesn't fit any of the above categories, it is considered a citation.

            **Output Format:**
            Return a JSON object with two top-level keys:
            - "reassured_documents": a dictionary where keys are the complete and confirmed Act titles and values are arrays of the correct relationship codes for that Act. An act can have multiple codes.
            - "missing_target_acts": a list of any target act titles from the list above that you could not find in the source text.

            **Example Response Format:**
            {{
                "reassured_documents": {{
                    "Privacy Act 2020": ["CIT", "AMD"],
                    "Employment Relations Act 2000": ["CIT"]
                }},
                "missing_target_acts": [
                    "Resource Management (Aquaculture Moratorium) Amendment Act 2004"
                ]
            }}

            If none of the proposed target acts are found in the text, return {{"reassured_documents": {{}}}} and set "missing_target_acts" to the full list of proposed targets.

            **CRITICAL:**
            - Only include acts that you can find in the text. Do not hallucinate.
            - Ensure the act titles in your response are complete and correct.
            - You can return multiple relationship codes for a single act if applicable, if it is cited, partially repealed or amended.
            - Every act from the proposed list must appear either under "reassured_documents" (with one or more relationship codes) or inside "missing_target_acts".
            """
            response = self.genai_client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "response_mime_type": "application/json"
                }
            )
            try:
                json_text = response.text.strip()
                # Handle potential markdown code block fences
                if json_text.startswith("```json"):
                    json_text = json_text[7:]
                if json_text.endswith("```"):
                    json_text = json_text[:-3]

                json_text = json_text.strip()
                return json.loads(json_text)
            except (json.JSONDecodeError, AttributeError) as e:
                print(f"Error parsing reassurance LLM response: {e}")
                raise

        return reassurance_llm_function
