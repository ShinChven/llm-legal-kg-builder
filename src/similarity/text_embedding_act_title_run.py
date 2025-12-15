from src.similarity.text_embedding_act_title import get_and_update_act_embeddings

act_titles_to_test = ["Estate and Gift Duties Amendment Act 1983"]

if __name__ == "__main__":
    # This is an example of how to use the function.
    # Make sure the act with this title exists in your database.
    print(f"Attempting to get and update embedding for: {act_titles_to_test[0]}")
    embeddings = get_and_update_act_embeddings(act_titles_to_test)

    if embeddings and act_titles_to_test[0] in embeddings:
        print(
            "Successfully received embedding of length: "
            f"{len(embeddings[act_titles_to_test[0]])}"
        )
    else:
        print("Failed to get or update embedding.")

