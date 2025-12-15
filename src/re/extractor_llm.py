import concurrent.futures
import json
import re
import time
from typing import Callable, Dict, List, Any, Set

from google import genai

from src.re.act_relationship_handler import ActRelationshipHandler
from src.re.normalization import normalize_title, normalize_text_for_search
# NOTE: Replaced structured logging (progress_log/error_log) with simple print statements per request.

# Data Models from former relationship_extraction.py
class Chunk(dict):
    pass

class FinalOutput(dict):
    pass


class DataParsingError(Exception):
    """Raised when all retries for parsing LLM responses fail for an act."""
    pass


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


class RelationshipExtractor:
    """
    Handles the extraction of legislative relationships, including recursive extraction.
    """
    def __init__(
        self,
        model_name: str,
        chunk_size: int,
        max_workers: int,
        act_handler: ActRelationshipHandler,
        use_cache: bool = True,
        enable_string_matching: bool = True,
    ):
        """Initialize extractor state and configuration."""
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.act_handler = act_handler
        self.use_cache = use_cache
        self.enable_string_matching = enable_string_matching
        self.genai_client = genai.Client()
        self.discovered_acts: Set[str] = set()
        self.acts_by_layer: Dict[int, Set[str]] = {}
        self.missing_acts: Set[str] = set()
        self.total_relationships = 0
        self.act_word_counts: Dict[str, int] = {}

    # --- Recursive Extraction Methods ---
    def run_recursive(self, core_act_title: str, layers: int):
        """Run recursive extraction starting from core_act_title for a number of layers."""
        print(f"Starting recursive extraction for '{core_act_title}' with {layers} layers.")

        # Reset statistics for a clean run
        self.discovered_acts = set()
        self.acts_by_layer = {}
        self.missing_acts = set()
        self.total_relationships = 0
        self.act_word_counts = {}

        normalized_core_title = normalize_title(core_act_title)
        self.discovered_acts.add(normalized_core_title)
        self.acts_by_layer[0] = {normalized_core_title}
        acts_for_current_layer = {normalized_core_title}

        for i in range(1, layers + 1):
            if not acts_for_current_layer:
                print(f"No more acts to process. Stopping at Layer {i}.")
                break
            found_acts = self._process_layer(acts_for_current_layer, i)
            newly_discovered_acts = found_acts - self.discovered_acts
            if newly_discovered_acts:
                self.acts_by_layer[i] = newly_discovered_acts
                self.discovered_acts.update(newly_discovered_acts)
            acts_for_current_layer = newly_discovered_acts

        print("Recursive relationship extraction process completed.")

    def _process_layer(self, acts_to_process: Set[str], current_layer: int) -> Set[str]:
        """
        Processes a single layer of relationship extraction.
        """
        all_found_acts = set()
        total_acts = len(acts_to_process)
        print(f"--- Starting Layer {current_layer} with {total_acts} act(s) ---")

        for i, act_title in enumerate(acts_to_process):
            print(f"-> Processing {i + 1}/{total_acts} in Layer {current_layer}: '{act_title}'")

            # Check for existing relationships in database if cache is enabled
            if self.use_cache:
                existing_relations = self.act_handler.get_relationships_by_subject(act_title)
                self.total_relationships += len(existing_relations)

                if existing_relations:
                    print(f"Found {len(existing_relations)} existing relationships for '{act_title}'. Using database records.")
                    for rel in existing_relations:
                        normalized_object_name = normalize_title(rel[2])
                        all_found_acts.add(normalized_object_name)
                    continue
                else:
                    print(f"No existing relationships found for '{act_title}'. Calling LLM.")
            else:
                print(f"Cache disabled. Calling LLM for '{act_title}'.")

            try:
                extraction_result = self._extract_for_single_act(
                    title=act_title,
                    enable_logging=True,
                    act_handler=self.act_handler
                )

                target_documents = extraction_result.get('target_documents', {})
                if not target_documents:
                    print(f"No new relationships found by LLM for '{act_title}'.")
                    continue

                # The cleaning is now done inside _extract_for_single_act.
                print(f"Storing {len(target_documents)} relationships for '{act_title}' in database.")
                for object_name, relationships in target_documents.items():
                    normalized_object_name = normalize_title(object_name)
                    self.act_handler.upsert_relationship(act_title, normalized_object_name, relationships)

                # After upserting, get the complete set of relationships from database
                all_db_relations = self.act_handler.get_relationships_by_subject(act_title)
                self.total_relationships += len(all_db_relations)
                print(f"Total relationships in database for '{act_title}': {len(all_db_relations)}")
                for rel in all_db_relations:
                    normalized_object_name = normalize_title(rel[2])
                    all_found_acts.add(normalized_object_name)

            except ValueError as e:
                print(f"Could not process '{act_title}': {e}")
                self.missing_acts.add(act_title)
            except Exception as e:
                print(f"An unexpected error occurred while processing '{act_title}': {e}")
            finally:
                # Mark the act as processed regardless of the outcome
                self.act_handler.mark_as_processed(act_title)
                print(f"Marked '{act_title}' as processed.")

        return all_found_acts

    def _clean_and_merge_target_documents(self, target_documents: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Cleans and merges target documents by checking for 'The ' prefix variations.

        It groups title variations (e.g., 'Some Act 2000' and 'The Some Act 2000').
        It then checks the database to see if the version with 'The ' is the official title.
        - If 'The {title}' exists in the database, all variations are merged under that title.
        - Otherwise, all variations are merged under the title without the prefix.
        """
        if not target_documents:
            return target_documents

        conn = self.act_handler.get_connection()
        if conn is None:
            print("Warning: Cannot verify titles against database. Proceeding without verification.")
            return target_documents

        try:
            # Group titles by their base form (without 'The ')
            title_groups: Dict[str, Dict[str, List[str]]] = {}
            for object_title, relationships in target_documents.items():
                normalized_title = normalize_title(object_title)
                base_title = normalized_title
                if normalized_title.lower().startswith('the '):
                    base_title = normalized_title[4:].strip()

                if base_title not in title_groups:
                    title_groups[base_title] = {}
                title_groups[base_title][normalized_title] = relationships

            cleaned_documents: Dict[str, List[str]] = {}
            with conn.cursor() as cursor:
                for base_title, variations in title_groups.items():
                    canonical_title = base_title

                    prefixed_title_to_check = f"The {base_title}"

                    cursor.execute(
                        "SELECT title FROM legislations WHERE LOWER(title) = LOWER(%s) LIMIT 1",
                        (prefixed_title_to_check,)
                    )
                    db_result = cursor.fetchone()

                    if db_result:
                        canonical_title = db_result[0]

                    merged_relationships = set()
                    for rels in variations.values():
                        merged_relationships.update(rels)

                    if canonical_title in cleaned_documents:
                        existing_rels = set(cleaned_documents[canonical_title])
                        existing_rels.update(merged_relationships)
                        cleaned_documents[canonical_title] = sorted(list(existing_rels))
                    else:
                        cleaned_documents[canonical_title] = sorted(list(merged_relationships))

                    if len(variations) > 1:
                        print(f"Merged variations {list(variations.keys())} into '{canonical_title}'.")
                    elif base_title != canonical_title:
                        print(f"Corrected title '{base_title}' to '{canonical_title}' based on database.")


            return cleaned_documents

        except Exception as e:
            print(f"Error during title cleaning and merging: {e}")
            return target_documents
        finally:
            if conn:
                self.act_handler.return_connection(conn)

    # --- Single Act Extraction Methods ---

    def _extract_for_single_act(
        self,
        title: str,
        enable_logging: bool = False,
        act_handler: ActRelationshipHandler = None
    ) -> FinalOutput:
        """
        Orchestrates the relationship extraction pipeline for a single legislative act.
        This is an internal method used by the recursive processor.
        """
        full_text = self._retrieve_legislation(title, act_handler)
        chunks = self._chunk_text(full_text)

        if enable_logging:
            total_words = len(full_text.split())
            total_chunks = len(chunks)
            print(f"Legislation word count: {total_words}")
            print(f"Chunk count: {total_chunks}")
            print(f"Processing {total_words} words in {total_chunks} chunks.")

        llm_function = self._create_llm_function(title, enable_logging)
        target_documents = self._process_chunks(chunks, llm_function, enable_logging)

        # --- Clean and merge target documents based on database verification ---
        if target_documents:
            original_count = len(target_documents)
            # The act_handler is passed from the parent recursive method
            target_documents = self._clean_and_merge_target_documents(target_documents)
            if enable_logging and original_count != len(target_documents):
                print(
                    f"Cleaned and merged {original_count} raw target documents down to {len(target_documents)}."
                )

        # --- Hallucination filtering: verify each target act title actually appears in source full text ---
        hallucinated_targets = []
        if target_documents:
            if self.enable_string_matching:
                # Prepare lowercase versions for case-insensitive search; also normalize punctuation
                normalized_full_text = normalize_text_for_search(full_text)
                for object_title in list(target_documents.keys()):
                    search_title = normalize_text_for_search(object_title)
                    if not search_title or search_title not in normalized_full_text:
                        # Move to hallucinated list and remove from real target documents
                        hallucinated_targets.append(object_title)
                        target_documents.pop(object_title, None)
                if enable_logging and hallucinated_targets:
                    print(
                        f"Filtered {len(hallucinated_targets)} hallucinated target act(s) (title not found in source text): "
                        + ", ".join(hallucinated_targets)
                    )
            elif enable_logging:
                print("String matcher disabled. Skipping hallucination filtering for this act.")

        # Filter out relationships where object Act year is newer than subject Act year
        subject_year = extract_act_year(title)
        if subject_year:
            filtered_target_documents = {}
            removed = []
            for object_title, rel_codes in target_documents.items():
                object_year = extract_act_year(object_title)
                # Only enforce if both have a detected year
                if object_year and object_year > subject_year:
                    removed.append((object_title, object_year))
                    continue
                filtered_target_documents[object_title] = rel_codes
            if removed and enable_logging:
                for (ot, oy) in removed:
                    print(f"Discarded relationship to '{ot}' (year {oy}) because it is newer than subject '{title}' ({subject_year}).")
            target_documents = filtered_target_documents

        # Remove self-referential relationship (object equals subject) if present
        normalized_subject = normalize_title(title)
        self_refs = [ot for ot in list(target_documents.keys()) if normalize_title(ot) == normalized_subject]
        if self_refs:
            for ot in self_refs:
                target_documents.pop(ot, None)
            if enable_logging:
                for ot in self_refs:
                    print(f"Removed self-referential relationship '{title}' -> '{ot}'.")

        # --- Upsert relationships to database for accumulative data aggregation ---
        if target_documents:
            if enable_logging:
                print(f"Upserting {len(target_documents)} relationships for '{title}' into the database.")
            for object_name, relationships in target_documents.items():
                normalized_object_name = normalize_title(object_name)
                # Use self.act_handler which is initialized with the class
                self.act_handler.upsert_relationship(title, normalized_object_name, relationships)

        final_output = FinalOutput({
            "source_title": title,
            "target_documents": dict(sorted(target_documents.items())),
            "hallucinated_targets": sorted(hallucinated_targets),
            "chunks": chunks,
        })
        return final_output

    def _retrieve_legislation(self, title: str, db_handler: ActRelationshipHandler) -> str:
        """
        Fetches the full text of a legislative document from the PostgreSQL database.
        """
        # Use handler's connection helper (singleton pool behind the scenes)
        conn = db_handler.get_connection()
        if conn is None:
            raise ConnectionError("Failed to get a database connection from the pool.")
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT text
                    FROM legislations
                    WHERE title = %s
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    (title,)
                )
                result = cursor.fetchone()
                if not result or not result[0]:
                    raise ValueError(
                        f"Legislation with title '{title}' not found in 'legislations' table or has no text."
                    )
                text = result[0]
                # Cache word count for processed act (subject role)
                try:
                    self.act_word_counts[normalize_title(title)] = len(text.split())
                except Exception:
                    # Fail silently; word count is auxiliary
                    pass
                return text
        except Exception as e:
            print(f"Error retrieving legislation '{title}': {e}")
            raise
        finally:
            db_handler.return_connection(conn)

    def _chunk_text(self, text: str) -> List[Chunk]:
        """
        Splits the full text of the legislation into manageable chunks.
        """
        if self.chunk_size > 1000000:
            print(f"Warning: Large chunk size ({self.chunk_size}) may exceed LLM token limits")

        # Detect if the text essentially lacks line breaks (e.g. scraped/minified HTML converted to plain text)
        newline_count = text.count('\n')
        if newline_count <= 2:
            # Fallback: sentence-preserving chunking to avoid losing context
            print(
                f"Fallback chunking strategy engaged: only {newline_count} line break(s) found. "
                "Splitting by sentences while respecting chunk_size (approx words)."
            )
            sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9\"\(\[])')
            raw_sentences = [s.strip() for s in sentence_pattern.split(text.strip()) if s.strip()]

            if not raw_sentences:
                return []

            chunks: List[Chunk] = []
            current_sentences: List[str] = []
            current_words = 0

            def flush():
                if current_sentences:
                    chunk_text = " ".join(current_sentences)
                    chunks.append(Chunk({"text": chunk_text, "section_ids": []}))

            for sentence in raw_sentences:
                sentence_word_count = len(sentence.split())

                # If single sentence itself exceeds chunk_size, we split it further by words (last resort)
                if sentence_word_count > self.chunk_size:
                    flush()
                    current_sentences.clear()
                    current_words = 0
                    words = sentence.split()
                    for i in range(0, len(words), self.chunk_size):
                        slice_words = words[i : i + self.chunk_size]
                        chunks.append(Chunk({"text": " ".join(slice_words), "section_ids": []}))
                    continue

                if current_words + sentence_word_count > self.chunk_size and current_sentences:
                    flush()
                    current_sentences = []
                    current_words = 0

                current_sentences.append(sentence)
                current_words += sentence_word_count

            flush()
            return chunks

        # Standard line-aware chunking
        lines = text.split('\n')
        chunks: List[Chunk] = []
        current_chunk_text = ""
        current_section_ids: List[str] = []
        current_word_count = 0

        section_pattern = re.compile(r"^\s*(\d+[A-Z]*)\.\s+.*")

        for line in lines:
            line_word_count = len(line.split())

            # If adding this line would exceed chunk size, flush current chunk first.
            if current_word_count + line_word_count > self.chunk_size and current_chunk_text:
                chunks.append(
                    Chunk({"text": current_chunk_text.rstrip(), "section_ids": list(set(current_section_ids))})
                )
                current_chunk_text = ""
                current_section_ids = []
                current_word_count = 0

            current_chunk_text += line + "\n"
            current_word_count += line_word_count

            match = section_pattern.match(line)
            if match:
                current_section_ids.append(match.group(1))

        if current_chunk_text:
            chunks.append(
                Chunk({"text": current_chunk_text, "section_ids": list(set(current_section_ids))})
            )

        return chunks

    def _process_chunks(
        self,
        chunks: List[Chunk],
        llm_function: Callable[[str, int], Dict],
        enable_logging: bool = False,
    ) -> Dict:
        """
        Orchestrates the extraction of target documents from text chunks in parallel.
        """
        all_target_documents = {}
        parse_failed = False

        def llm_task(chunk: Chunk, index: int, total_chunks: int):
            if enable_logging:
                word_count = len(chunk["text"].split())
                print(f"Starting chunk {index + 1}/{total_chunks} ({word_count} words)...")

            max_retries = 10
            delay_seconds = 10
            for attempt in range(1, max_retries + 1):
                try:
                    result = llm_function(chunk["text"], total_chunks)
                    if enable_logging:
                        print(f"Completed chunk {index + 1}/{total_chunks} on attempt {attempt}")
                    return result
                except Exception as exc:
                    if attempt < max_retries:
                        if enable_logging:
                            print(
                                f"Chunk {index + 1}/{total_chunks} attempt {attempt} failed: {exc}. "
                                f"Retrying in {delay_seconds}s..."
                            )
                        time.sleep(delay_seconds)
                    else:
                        if enable_logging:
                            print(
                                f"Chunk {index + 1}/{total_chunks} failed after {max_retries} attempts: {exc}. "
                                "Marking act extraction as failed."
                            )
                        return {"__failed__": True}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            total_chunks = len(chunks)
            futures = {
                executor.submit(llm_task, chunk, i, total_chunks)
                for i, chunk in enumerate(chunks)
            }

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if not result:
                    continue
                if "__failed__" in result:
                    parse_failed = True
                    continue
                if "target_documents" in result:
                    for title, relationship_codes in result["target_documents"].items():
                        if title not in all_target_documents:
                            all_target_documents[title] = set()
                        all_target_documents[title].update(relationship_codes)
        if parse_failed:
            # If any chunk failed all retries, treat the whole act extraction as failed
            raise DataParsingError("One or more chunks failed all retry attempts.")
        return {title: list(codes) for title, codes in all_target_documents.items()}

    def _create_llm_function(self, legislation_title: str, enable_logging: bool = False) -> Callable[[str, int], Dict]:
        """
        Creates the LLM function to be used for processing chunks.
        """
        def llm_function(text_chunk: str, total_chunks: int) -> Dict:
            if total_chunks > 1:
                passage_info = f"a chunk from the act '{legislation_title}'"
            else:
                passage_info = f"the full text of the act '{legislation_title}'"

            prompt = f"""
            You are an expert legal analyst specializing in Legislation, tasked with extracting relationships between legislative documents.
            Current Region is New Zealand. You will use your knowledge of New Zealand legislation to inform your analysis.

            The following text is {passage_info}:
            ```
            {text_chunk}
			```
            Extract legislative relationships from the text and return target documents with their relationship types.

            IMPORTANT: Only extract relationships where the target is a complete Act title, such as:
            - Acts (e.g., "Social Security Act 2018", "Employment Relations Act 2000", "Privacy Act 2020")

            CRITICAL: Before including any relationship in your response, think twice and verify that:
            1. The target is specifically an "Act" (not regulations, rules, orders, bills, or other types of legislation)
            2. The target has a complete Act title with a year (e.g., "Privacy Act 2020")
            3. You are not confusing references to regulations, statutory instruments, or other non-Act legislation
            4. Try your best to identify as many Acts as possible.

            The relationship types are:
			- AMD (Amendment) - when the source document amends another Act
    		- PRP (Partial Repeal) - when the source document partially repeals one or more parts of another Act
    		- FRP (Full Repeal) - when the source document completely repeals another Act
    		- CIT (Citation) - when you can't decide the relationship or it doesn't fit any of the above categories, it is considered a citation.

            Return a JSON object with "target_documents" key containing a dictionary where:
            - Keys are the complete Act titles
            - Values are arrays of relationship codes for that Act

            Example response format:
            {{
                "target_documents": {{
                    "Privacy Act 2020": ["CIT", "AMD"],
                    "Employment Relations Act 2000": ["CIT"]
                }}
            }}

            If no relationships to Acts are found, return {{"target_documents": {{}}}}.

            **Be cautious, some of the Acts may have very similar titles, only the year in the title is different. You should treat them as separate Acts.**

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
                if json_text.startswith("```json"):
                    json_text = json_text[7:]
                if json_text.endswith("```"):
                    json_text = json_text[:-3]

                json_text = json_text.strip()
                return json.loads(json_text)
            except (json.JSONDecodeError, AttributeError) as e:
                if enable_logging:
                    print(f"Could not decode llm response to json (will trigger retry): {e}")
                    if hasattr(response, 'text'):
                        print(f"Raw response: {response.text}")
                # Raise to allow retry logic in _process_chunks
                raise

        return llm_function

    # --- Tree Building Methods ---

    def build_relationship_tree(self, core_act_title: str, layers: int) -> Dict[str, Any]:
        """
        Builds a nested dictionary representing the relationship tree.
        """
        normalized_core_title = normalize_title(core_act_title)
        print(f"Building relationship tree for '{normalized_core_title}' up to {layers} layers deep.")
        root_children = self._build_node(normalized_core_title, 1, layers)
        tree = {
            normalized_core_title: {
                "children": root_children,
                "word_count": self.act_word_counts.get(normalized_core_title),
            }
        }
        return tree

    def _build_node(self, act_title: str, current_layer: int, max_layers: int) -> Dict[str, Any]:
        """
        Recursively builds a node and its children for the tree.
        """
        if current_layer > max_layers:
            return {}

        normalized_act_title = normalize_title(act_title)
        relationships = self.act_handler.get_relationships_by_subject(normalized_act_title)
        if not relationships:
            return {}

        children_dict = {}
        for rel in relationships:
            object_name = normalize_title(rel[2])
            relationship_types = rel[3]

            grandchildren = self._build_node(object_name, current_layer + 1, max_layers)

            node_details = {
                "relationships": relationship_types,
                "word_count": self.act_word_counts.get(object_name),
            }
            if grandchildren:
                node_details["children"] = grandchildren

            children_dict[object_name] = node_details

        return children_dict

    def get_statistics(self) -> Dict[str, Any]:
        """
        Calculates and returns statistics about the extraction process.
        """
        # Directly use self.act_handler for statistics
        conn = None
        try:
            conn = self.act_handler.get_connection()
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(DISTINCT title) FROM legislations")
                total_acts_in_db = cursor.fetchone()[0]
        except Exception as e:
            print(f"Could not retrieve total acts from database: {e}")
            total_acts_in_db = -1  # Indicate error
        finally:
            if conn is not None:
                try:
                    self.act_handler.return_connection(conn)
                except Exception:
                    pass

        coverage = (
            (len(self.discovered_acts) / total_acts_in_db * 100)
            if total_acts_in_db > 0
            else 0
        )

        stats = {
            "total_unique_acts_discovered": len(self.discovered_acts),
            "total_relationships_found": self.total_relationships,
            "coverage_of_db": f"{coverage:.2f}%",
            "acts_per_layer": {
                f"layer_{i}": len(acts) for i, acts in self.acts_by_layer.items()
            },
            "missing_acts_not_in_db": sorted(list(self.missing_acts)),
        }
        return stats
