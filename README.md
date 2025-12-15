# NZ Legislation Relationship Extraction & Knowledge Graph

This project is a comprehensive framework for building a **Legal Knowledge Graph** from New Zealand legislation. It utilizes **Large Language Models (LLMs)** (specifically Google Gemini via `google-genai`) to extract citations and relationships between Acts, constructing a rich graph database for analysis.

The system moves beyond simple keyword matching, employing generative AI to understand the context of references (e.g., "amends", "repeals", "cited by"). It then leverages graph algorithms (Leiden Community Detection, PageRank) to identify clusters of related legislation and influential Acts.

## 🌟 Key Features

*   **LLM-Driven Extraction:** Uses Google's Gemini models to parse legal text and identify complex relationships between statutes.
*   **Hybrid Graph Construction:** Combines structured XML data (where available) with unstructured text extraction.
*   **Graph Analytics:**
    *   **Community Detection:** Implements the Leiden algorithm to discover thematic clusters of legislation.
    *   **Centrality Measures:** Calculates Weighted PageRank and Katz Centrality to find key nodes.
*   **Topic Modeling:** Generates topics for legislative communities using LLMs.
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

*   **Python 3.9+**
*   **PostgreSQL** (for storing raw texts and extracted relationships)
*   **Neo4j** (for the knowledge graph)
*   **Google Gemini API Key** (for LLM extraction)

### Installation

This project manages dependencies using [Poetry](https://python-poetry.org/).

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/nzlegislation-subscribe.git
    cd nzlegislation-subscribe
    ```

2.  **Install dependencies:**
    ```bash
    poetry install
    ```

3.  **Environment Setup:**
    Create a `.env` file in the root directory (see `.env.example` if available) and add your credentials:
    ```env
    # Database Config
    POSTGRES_DB=...
    POSTGRES_USER=...
    POSTGRES_PASSWORD=...
    POSTGRES_HOST=localhost
    
    # Neo4j Config
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=...
    
    # LLM Config
    GOOGLE_API_KEY=...
    ```

## 📖 Usage

### 1. Data Ingestion & Cleaning
Scripts located in `src/datasets/` handle the downloading and normalization of legislative text.

```bash
# Example: Run the scraper (check specific script arguments)
python -m src.datasets.scraper
```

### 2. Relationship Extraction
The core extraction logic resides in `src/re/`.

**Configuration:**
Before running, ensure you have set the `CORE_ACT` variable in your `.env` file to the specific Act title you wish to process (or leave it blank/configure the script to process all).

```bash
# Run LLM-based extraction
python -m src.re.extractor_llm_run
```

### 3. Graph Construction
Once relationships are stored in Postgres, build the Neo4j graph:

```bash
python -m src.graph.neo4j_graph_construction
```

### 4. Graph Analytics
Run algorithms to analyze the graph structure:

```bash
# Run Leiden Community Detection
python -m src.graph.leiden_community_detection

# Calculate PageRank
python -m src.graph.weighted_page_rank
```

### 5. Topic Modeling
Generate topics for the discovered communities:

```bash
python -m src.topic.run
```

### 6. Visualization
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
*   `src/similarity/`: Vector embedding and similarity search.
*   `src/topic/`: Topic modeling and categorization.
*   `outputs/`: Generated graphs, plots, and reports.

## 📄 License

[License Information Here]
