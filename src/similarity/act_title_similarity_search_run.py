from src.similarity.act_title_similarity_search import find_similar_acts

if __name__ == "__main__":
    search_title = "Acts and Regulations Publications Act 1989"
    similarity_threshold = 0.98

    print(f"Searching for acts similar to: '{search_title}' with a threshold of {similarity_threshold}")
    similar_acts = find_similar_acts(search_title, similarity_threshold)

    if similar_acts:
        print("Found similar acts:")
        for act in similar_acts:
            print(f"  - Title: {act['title']}, Similarity: {act['similarity']:.4f}")
    else:
        print("No similar acts found.")
