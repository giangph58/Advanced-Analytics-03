import pandas as pd
import numpy as np
from pyspark.sql.functions import pandas_udf, col, avg, stddev, round, concat_ws
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

# Arxiv high-level category mapping
arxiv_main_category = {
    "astro-ph": [
        "astro-ph.CO", "astro-ph.EP", "astro-ph.GA", "astro-ph.HE",
        "astro-ph.IM", "astro-ph.SR"
    ],
    "cond-mat": [
        "cond-mat.dis-nn", "cond-mat.mes-hall", "cond-mat.mtrl-sci",
        "cond-mat.other", "cond-mat.quant-gas", "cond-mat.soft",
        "cond-mat.stat-mech", "cond-mat.str-el", "cond-mat.supr-con"
    ],
    "gr-qc": [
        "gr-qc"
    ],
    "hep": [
        "hep-ex", "hep-lat", "hep-ph", "hep-th"
    ],
    "math-ph": [
        "math-ph"
    ],
    "nlin": [
        "nlin.AO", "nlin.CG", "nlin.CD", "nlin.SI", "nlin.PS"
    ],
    "nucl": [
        "nucl-ex", "nucl-th"
    ],
    "quant-ph": [
        "quant-ph"
    ],
    "physics": [
        "physics.acc-ph", "physics.ao-ph", "physics.app-ph", "physics.atm-clus",
        "physics.atom-ph", "physics.bio-ph", "physics.chem-ph", "physics.class-ph",
        "physics.comp-ph", "physics.data-an", "physics.ed-ph", "physics.flu-dyn",
        "physics.gen-ph", "physics.geo-ph", "physics.hist-ph", "physics.ins-det",
        "physics.med-ph", "physics.optics", "physics.plasm-ph", "physics.pop-ph",
        "physics.soc-ph", "physics.space-ph"
    ],
    "math": [
        "math.AC", "math.AG", "math.AP", "math.AT", "math.CA", "math.CO",
        "math.CT", "math.CV", "math.DG", "math.DS", "math.FA", "math.GM",
        "math.GN", "math.GR", "math.GT", "math.HO", "math.IT", "math.KT",
        "math.LO", "math.MG", "math.MP", "math.NA", "math.NT", "math.OA",
        "math.OC", "math.PR", "math.QA", "math.RA", "math.RT", "math.SG",
        "math.SP", "math.ST"
    ],
    "cs": [
        "cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR", "cs.CV",
        "cs.CY", "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS", "cs.ET", "cs.FL",
        "cs.GL", "cs.GR", "cs.GT", "cs.HC", "cs.IR", "cs.IT", "cs.LG", "cs.LO",
        "cs.MA", "cs.MM", "cs.MS", "cs.NA", "cs.NE", "cs.NI", "cs.OH", "cs.OS",
        "cs.PF", "cs.PL", "cs.RO", "cs.SC", "cs.SD", "cs.SE", "cs.SI", "cs.SY"
    ],
    "q-bio": [
        "q-bio.BM", "q-bio.CB", "q-bio.GN", "q-bio.MN", "q-bio.NC", "q-bio.OT",
        "q-bio.PE", "q-bio.QM", "q-bio.SC", "q-bio.TO"
    ],
    "q-fin": [
        "q-fin.CP", "q-fin.EC", "q-fin.GN", "q-fin.MF", "q-fin.PM", "q-fin.PR",
        "q-fin.RM", "q-fin.ST", "q-fin.TR"
    ],
    "stat": [
        "stat.AP", "stat.CO", "stat.ME", "stat.ML", "stat.OT", "stat.TH"
    ],
    "eess": [
        "eess.AS", "eess.IV", "eess.SP", "eess.SY"
    ],
    "econ": [
        "econ.EM", "econ.GN", "econ.TH"
    ],
}

category_to_parent = {child: parent for parent, children in arxiv_main_category.items() for child in children}
# Define a Python function for mapping (no need for UDF in pandas)
def map_category(category):
    return category_to_parent.get(category, "other")

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