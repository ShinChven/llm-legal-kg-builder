# NZ Legislation Relationship Extraction & Knowledge Graph

This is an LLM-driven ensemble information extraction framework to construct legislation knowledge graphs.

The system moves beyond simple keyword matching, employing generative AI to understand the context of references (e.g., "amends", "repeals", "cited by") and to perform topic modeling and classification. It then leverages graph algorithms (Leiden Community Detection, PageRank) to identify clusters of related legislation and influential Acts.

## 🌟 Key Features

*   **LLM-Driven Extraction:** Uses Google's Gemini models to parse legal text and identify complex relationships between statutes.
*   **Hybrid Graph Construction:** Combines structured XML data (where available) with unstructured text extraction.
*   **Graph Analytics:**
    *   **Community Detection:** Implements the Leiden algorithm to discover thematic clusters of legislation.
    *   **Centrality Measures:** Calculates Weighted PageRank and Katz Centrality to find key nodes.
*   **Topic Classification:** Classifies legislation into official **Parliamentary Select Committee** categories and subject areas using LLMs. These classifications are aggregated to analyze the thematic composition of legislative communities.
*   **Visualization:** Includes tools for 3D graph visualization and static statistical plots.
*   **Dual-Database Architecture:** Uses **PostgreSQL** for relational storage and **Neo4j** for graph traversals.

## 🛠️ Architecture

The pipeline consists of four main stages:

1.  **Ingestion:** Scraping and indexing legislation from NZLII and legislation.govt.nz.
2.  **Extraction (RE):** Processing text chunks with LLMs to extract `(Subject Act) -> [Relationship] -> (Object Act)` triples.
3.  **Graph Build:** Projecting extracted relationships into a Neo4j graph database.
4.  **Analysis:** Running graph algorithms and topic modeling to derive insights.

## 🚀 Getting Started

### Prerequisites

*   **Python 3.12+**
*   **PostgreSQL** (for storing raw texts and extracted relationships)
*   **Neo4j** (for the knowledge graph)
*   **Google Gemini API Key** (for LLM extraction)

## 📖 Usage

### 1. Relationship Extraction
The core extraction logic resides in `src/re/`.

*   `src/re/extractor_llm.py`: The core LLM-based relationship extraction module.
*   `src/re/extractor_llm_run.py`: A **single-shot** runner that processes documents once for simple extraction.
*   `src/re/extractor_llm_run_heavy.py`: An **ensemble** runner that employs a multi-pass strategy (NER followed by Relationship Extraction verification) for high-accuracy information extraction.

**Configuration:**
Before running, ensure you have set the `CORE_ACT` variable in your `.env` file to the specific Act title you wish to process (or leave it blank/configure the script to process all).

```bash
# Run LLM-based extraction (single-shot, faster)
python -m src.re.extractor_llm_run

# Run LLM-based extraction (ensemble, high accuracy)
python -m src.re.extractor_llm_run_heavy
```

### 2. Graph Construction
Once relationships are stored in Postgres, build the Neo4j graph:

```bash
python -m src.graph.neo4j_graph_construction
```

### 3. Graph Analytics
Run algorithms to analyze the graph structure:

```bash
# Run Leiden Community Detection
python -m src.graph.leiden_community_detection

# Calculate PageRank
python -m src.graph.weighted_page_rank
```

### 4. Topic Classification
Classify Acts into Parliamentary Select Committee topics.

```bash
# Run topic classification for a single Act
# Set TOPIC_ACT (or CORE_ACT) in .env or via environment variable
TOPIC_ACT="Income Tax Act 2007" python -m src.topic.topic_extractor_llm_run

# Run batch classification for all unprocessed Acts
python -m src.topic.topic_extractor_llm_run_batch
```

### 5. Visualization
Generate charts and visualizations in `outputs/`:

```bash
python -m src.analytics.visualize_relationships_static
```

## 📂 Project Structure

*   `src/analytics/`: Visualization and statistical analysis scripts.
*   `src/datasets/`: Scrapers, cleaners, and data loaders.
*   `src/db/`: Database schemas and connection logic (Postgres).
*   `src/evaluate/`: Evaluation metrics for extraction quality.
*   `src/graph/`: Neo4j interaction and graph algorithms.
*   `src/re/`: Relationship Extraction (LLM & XML based).
    *   `extractor_llm.py`: Core module for LLM-based relationship extraction.
    *   `extractor_llm_run.py`: Single-shot script for simple, one-pass extraction.
    *   `extractor_llm_run_heavy.py`: Ensemble script for high-accuracy, multi-pass extraction (NER + verification).
*   `src/similarity/`: Vector embedding and similarity search.
*   `src/topic/`: Topic Classification (Select Committees).
    *   `topic_extractor_llm.py`: Core module for classifying Acts into committee topics.
    *   `topic_extractor_llm_run.py`: Runner for single-Act classification.
    *   `topic_extractor_llm_run_batch.py`: Runner for batch classification of all Acts.
    *   `categories.py`: Definitions of Parliamentary Select Committees and their topics.
*   `outputs/`: Generated graphs, plots, and reports.

## 📄 License

[License Information Here]
