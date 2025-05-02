import pandas as pd
import numpy as np
from pyspark.sql.functions import pandas_udf, col, avg, stddev, round, concat_ws
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

# Load a sentence embedding model (e.g., all-MiniLM-L6-v2)
embedder = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2", tokenizer="sentence-transformers/all-MiniLM-L6-v2")


def analyze_similarity(df, text_feature, target, output_column=None, num_rows=20):
    """
    Analyze the embedding similarity between text and category columns.
    
    Parameters:
    -----------
    df : DataFrame
        The input Spark DataFrame
    text_feature : str
        Name of the column containing the text content (e.g. "summary", "title")
    target : str
        Name of the column containing the category (e.g. "main_category", "arxiv_main_category")
    output_column : str, optional
        Name for the output similarity column. If None, it will be auto-generated.
    num_rows : int, optional
        Number of rows to display in the results, default 20
        
    Returns:
    --------
    tuple
        (df_with_similarity, category_stats_df)
    """
    # Generate output column name if not provided
    if output_column is None:
        output_column = f"{text_feature}_{target}_similarity"
    
    # Use the same similarity function
    @pandas_udf("float")
    def compute_similarity(text_feature: pd.Series, category: pd.Series) -> pd.Series:
        sim_scores = []
        for t, c in zip(text_feature, category):
            # Handle texts longer than model's max length
            emb_feat = np.mean(embedder(t, truncation=True, max_length=512)[0], axis=0).reshape(1, -1)
            emb_cat = np.mean(embedder(c, truncation=True, max_length=512)[0], axis=0).reshape(1, -1)
            sim = cosine_similarity(emb_feat, emb_cat)[0][0]
            sim_scores.append(sim)
        return pd.Series(sim_scores)
    
    # Compute similarity and cache the result
    print(f"Computing similarity between '{text_feature}' and '{target}'...")
    df_with_similarity = df.withColumn(
        output_column, 
        compute_similarity(col(text_feature), col(target))
    ).cache()
    
    # Compute statistics by category
    similarity_stats = df_with_similarity.groupBy(target).agg(
        round(avg(output_column), 4).alias("avg_similarity"),
        round(stddev(output_column), 4).alias("stddev_similarity")
    )
    
    # Format with ± notation
    similarity_stats = similarity_stats.withColumn(
        "avg ± std",
        concat_ws(" ± ", col("avg_similarity"), col("stddev_similarity"))
    )
    
    # Display results sorted by similarity
    print(f"\nCategory similarity statistics (sorted by avg similarity):")
    similarity_stats.orderBy(col("avg_similarity").desc()).show(num_rows, truncate=False)
    
    # Return both DataFrames for further analysis
    return None