from psycopg2.extras import Json
from src.db.db_connection import db_connection
from src.db.create_act_relationships_table import ensure_act_relationships_table


class ActRelationshipHandler:
    def __init__(self):
        """Uses the global singleton db_connection; no parameter injection."""
        self.db_conn = db_connection

    # --- Connection convenience API (so other modules don't need a shim) ---
    def get_connection(self):
        return self.db_conn.get_connection()

    # Some legacy code expected a method named return_connection; keep both.
    def release_connection(self, conn):
        self.db_conn.release_connection(conn)

    def return_connection(self, conn):  # alias
        self.release_connection(conn)

    def _ensure_table(self):
        """Ensure the act_relationships table exists (idempotent)."""
        ensure_act_relationships_table()

    def upsert_relationship(self, subject_name: str, object_name: str, relationships: list):
        """
        Inserts a new relationship or updates an existing one by merging relationships.
        """
        self._ensure_table()
        conn = self.db_conn.get_connection()
        if conn is None:
            print("PostgreSQL connection not available. Cannot upsert relationship.")
            return

        try:
            with conn.cursor() as cursor:
                # First, check if relationship already exists to provide better logging
                cursor.execute(
                    "SELECT relationships FROM act_relationships WHERE subject_name = %s AND object_name = %s",
                    (subject_name, object_name)
                )
                existing = cursor.fetchone()

                cursor.execute(
                    """
                    INSERT INTO act_relationships (subject_name, object_name, relationships)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (subject_name, object_name)
                    DO UPDATE SET
                        relationships = (
                            SELECT jsonb_agg(DISTINCT elem ORDER BY elem)
                            FROM (
                                SELECT jsonb_array_elements_text(act_relationships.relationships) AS elem
                                UNION
                                SELECT jsonb_array_elements_text(EXCLUDED.relationships) AS elem
                            ) combined
                        ),
                        updated_at = CURRENT_TIMESTAMP;
                    """,
                    (subject_name, object_name, Json(relationships)),
                )

                if existing:
                    existing_rels = existing[0] if existing[0] else []
                    combined_rels = list(set(existing_rels + relationships))
                    print(f"Merged relationships: {subject_name} -> {object_name}")
                    print(f"  Previous: {existing_rels}")
                    print(f"  New: {relationships}")
                    print(f"  Combined: {sorted(combined_rels)}")
                else:
                    print(f"Created new relationship: {subject_name} -> {object_name} with {relationships}")

                conn.commit()
        except Exception as e:
            print(f"Error upserting relationship for {subject_name} -> {object_name}: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                self.db_conn.release_connection(conn)

    def get_word_count_for_act(self, act_title: str) -> int:
        """
        Retrieves the word count for a specific act from the legislations table.
        """
        conn = self.db_conn.get_connection()
        if conn is None:
            print("PostgreSQL connection not available. Cannot get word count.")
            return 0

        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT word_count FROM legislations
                    WHERE title = %s;
                    """,
                    (act_title,),
                )
                result = cursor.fetchone()
                return result[0] if result else 0
        except Exception as e:
            print(f"Error getting word count for act {act_title}: {e}")
            return 0
        finally:
            if conn:
                self.db_conn.release_connection(conn)

    def update_relationship_codes(self, subject_name: str, object_name: str, new_relationships: list):
        """
        Updates an existing relationship record with a new set of relationship codes, overwriting the old ones.
        """
        self._ensure_table()
        conn = self.db_conn.get_connection()
        if conn is None:
            print("PostgreSQL connection not available. Cannot update relationship codes.")
            return

        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE act_relationships
                    SET
                        relationships = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE subject_name = %s AND object_name = %s;
                    """,
                    (Json(new_relationships), subject_name, object_name)
                )
                conn.commit()
                if cursor.rowcount > 0:
                    print(f"Updated relationship codes for {subject_name} -> {object_name} to {new_relationships}")
                else:
                    # This case might happen if a relationship was deleted mid-process, which is unlikely but good to log.
                    print(f"Warning: No relationship found for {subject_name} -> {object_name} to update.")

        except Exception as e:
            print(f"Error updating relationship codes for {subject_name} -> {object_name}: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                self.db_conn.release_connection(conn)

    def delete_relationship(self, subject_name: str, object_name: str) -> bool:
        """
        Removes a specific relationship record entirely.

        Returns True if a row was deleted, False otherwise.
        """
        self._ensure_table()
        conn = self.db_conn.get_connection()
        if conn is None:
            print("PostgreSQL connection not available. Cannot delete relationship.")
            return False

        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM act_relationships
                    WHERE subject_name = %s AND object_name = %s;
                    """,
                    (subject_name, object_name)
                )
                conn.commit()
                if cursor.rowcount > 0:
                    print(f"Deleted relationship: {subject_name} -> {object_name}")
                    return True
                print(f"No relationship found to delete for {subject_name} -> {object_name}.")
                return False
        except Exception as e:
            print(f"Error deleting relationship for {subject_name} -> {object_name}: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                self.db_conn.release_connection(conn)

    def get_relationship(self, subject_name: str, object_name: str):
        """
        Retrieves a specific relationship.
        """
        self._ensure_table()
        conn = self.db_conn.get_connection()
        if conn is None:
            print("PostgreSQL connection not available. Cannot get relationship.")
            return None

        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT * FROM act_relationships
                    WHERE subject_name = %s AND object_name = %s;
                    """,
                    (subject_name, object_name),
                )
                return cursor.fetchone()
        except Exception as e:
            print(f"Error getting relationship for {subject_name} -> {object_name}: {e}")
            return None
        finally:
            if conn:
                self.db_conn.release_connection(conn)

    def get_relationships_by_subject(self, subject_name: str):
        """
        Retrieves all relationships for a given subject.
        """
        self._ensure_table()
        conn = self.db_conn.get_connection()
        if conn is None:
            print("PostgreSQL connection not available. Cannot get relationships by subject.")
            return []

        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT * FROM act_relationships
                    WHERE subject_name = %s;
                    """,
                    (subject_name,),
                )
                return cursor.fetchall()
        except Exception as e:
            print(f"Error getting relationships for subject {subject_name}: {e}")
            return []
        finally:
            if conn:
                self.db_conn.release_connection(conn)

    def delete_relationships_by_subject(self, subject_name: str) -> int:
        """Delete all relationships where the given act is the subject.

        Returns number of deleted rows (best-effort; -1 on error).
        """
        self._ensure_table()
        conn = self.db_conn.get_connection()
        if conn is None:
            print("PostgreSQL connection not available. Cannot delete relationships by subject.")
            return -1

        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM act_relationships
                    WHERE subject_name = %s;
                    """,
                    (subject_name,),
                )
                deleted = cursor.rowcount
                conn.commit()
                print(f"Deleted {deleted} existing relationship rows for subject '{subject_name}'.")
                return deleted
        except Exception as e:
            print(f"Error deleting relationships for subject {subject_name}: {e}")
            if conn:
                conn.rollback()
            return -1
        finally:
            if conn:
                self.db_conn.release_connection(conn)

    def get_relationship_layers(self, subject_name: str, num_layers: int):
        """
        Retrieves a specified number of layers of relationships for a given subject,
        returning a nested tree structure.
        """
        self._ensure_table()
        if num_layers <= 0:
            return {}

        # The root of our result tree
        tree = {"act": subject_name, "children": []}

        # The queue will store tuples of:
        # (subject_name_to_process, parent_node_in_tree_to_attach_to, current_depth)
        queue = [(subject_name, tree, 0)]

        # Visited helps prevent cycles and redundant processing
        visited = {subject_name}

        while queue:
            current_subject, parent_node, current_layer = queue.pop(0)

            if current_layer >= num_layers:
                continue

            relationships = self.get_relationships_by_subject(current_subject)

            for rel in relationships:
                # rel is a tuple: (id, subject_name, object_name, relationships, ...)
                object_name = rel[2]
                relationship_types = rel[3]

                # Create the child node for the tree
                child_node = {
                    "act": object_name,
                    "relationship": relationship_types,
                    "children": []
                }

                # Attach the new child to its parent in the tree
                parent_node["children"].append(child_node)

                # If we haven't visited this new node before, add it to the queue
                # to process its own children in the next layer.
                if object_name not in visited:
                    visited.add(object_name)
                    queue.append((object_name, child_node, current_layer + 1))

        # Recursively prune empty "children" lists from the tree for cleaner output
        def prune_empty_children(node):
            if "children" in node and node["children"]:
                for child in list(node["children"]):
                    prune_empty_children(child)
                if not node["children"]:
                    del node["children"]
            elif "children" in node:
                 del node["children"]

        prune_empty_children(tree)
        return tree

    def get_network_statistics(self, relationship_tree: dict, num_layers_requested: int):
        """
        Analyzes a relationship tree and returns a dictionary of statistics.
        """
        if not relationship_tree or 'act' not in relationship_tree:
            return {}

        core_act = relationship_tree['act']
        all_acts = {core_act}
        # Count individual relationship codes (not just edges/rows)
        total_relationships = 0
        # Unique acts first discovered at each layer (root at 0)
        acts_per_layer = {0: 1}
        # Number of relationship codes that connect into each layer (1..N)
        relationships_per_layer = {}
        max_depth = 0

        queue = [(relationship_tree, 0)]

        while queue:
            node, layer = queue.pop(0)

            # Avoid traversing beyond requested depth
            if layer >= num_layers_requested:
                continue

            children = node.get("children", [])

            next_layer = layer + 1
            if next_layer not in acts_per_layer:
                acts_per_layer[next_layer] = 0

            for child in children:
                child_act = child['act']
                rel_codes = child.get('relationship', []) or []
                rel_count = len(rel_codes)

                # Count relationship codes per child edge into the next layer
                if rel_count:
                    relationships_per_layer[next_layer] = relationships_per_layer.get(next_layer, 0) + rel_count
                    total_relationships += rel_count
                else:
                    # If no codes present, still treat it as a single generic edge
                    relationships_per_layer[next_layer] = relationships_per_layer.get(next_layer, 0) + 1
                    total_relationships += 1

                # Track unique acts discovered at this layer
                if child_act not in all_acts:
                    all_acts.add(child_act)
                    acts_per_layer[next_layer] += 1

                # Enqueue child to ensure depth accounting, even if it has no children
                queue.append((child, next_layer))
                max_depth = max(max_depth, next_layer)

        # --- Database-related stats ---
        conn = self.db_conn.get_connection()
        if conn is None:
            print("PostgreSQL connection not available for statistics.")
            found_in_db_count = "N/A"
            missing_from_db_count = "N/A"
            total_legislations_in_db = "N/A"
        else:
            try:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT title FROM legislations;")
                    all_db_titles = {row[0] for row in cursor.fetchall()}
                    total_legislations_in_db = len(all_db_titles)

                    found_acts_in_db = all_acts.intersection(all_db_titles)
                    found_in_db_count = len(found_acts_in_db)
                    missing_from_db_count = len(all_acts) - found_in_db_count
            except Exception as e:
                print(f"Error fetching legislation titles for stats: {e}")
                found_in_db_count = "Error"
                missing_from_db_count = "Error"
                total_legislations_in_db = "Error"
            finally:
                self.db_conn.release_connection(conn)

        return {
            "core_act": core_act,
            "layers_requested": num_layers_requested,
            "layers_found": max_depth,
            "total_relationships": total_relationships,
            "total_acts": len(all_acts),
            "acts_found_in_db": found_in_db_count,
            "acts_missing_from_db": missing_from_db_count,
            "total_legislations_in_db": total_legislations_in_db,
            "acts_per_layer": acts_per_layer,
            "relationships_per_layer": relationships_per_layer,
        }

    def mark_as_processed(self, title: str):
        """
        Marks a legislation as processed by setting the processed_at timestamp.
        """
        conn = self.db_conn.get_connection()
        if conn is None:
            print("PostgreSQL connection not available. Cannot mark as processed.")
            return

        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE legislations
                    SET processed_at = CURRENT_TIMESTAMP
                    WHERE title = %s;
                    """,
                    (title,)
                )
                conn.commit()
        except Exception as e:
            print(f"Error marking '{title}' as processed: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                self.db_conn.release_connection(conn)
