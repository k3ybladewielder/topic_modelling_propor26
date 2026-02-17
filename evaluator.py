# funcao para avaliacao
from gensim import corpora
from gensim.models import CoherenceModel
from tqdm import tqdm
import numpy as np
import pandas as pd


# abrindo dfs
base_path = "../results/df_topics_*.parquet"
files = sorted(glob.glob(base_path))
dfs = []

for f in files:
    test_k = f.split("df_topics_")[1].split(".")[0]
    df_temp = pd.read_parquet(f)
    df_temp["test_k"] = test_k
    dfs.append(df_temp)
    print(f"Carregado {f} com test_k={test_k} ({len(df_temp)} linhas)")

df_all_topics =  pd.concat(dfs, ignore_index=True)
del dfs



import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def evaluate_topic_coherence(df_topics, text_col="news_content_clean", topic_col="topic",
                             method_col="method", round_col="round",
                             metrics=("c_v", "c_npmi"), topn=10):
    """
    Avalia a coerência dos tópicos por método e rodada usando métricas C_V e C_NPMI.
    Ignora tópicos vazios e trata casos com poucos documentos.
    """

    results = []
    df_topics["_tokens_"] = df_topics[text_col].apply(lambda x: str(x).split())

    for (method, rnd), group in tqdm(df_topics.groupby([method_col, round_col]), desc="Avaliando coerência"):
        try:
            tokenized_docs = group["_tokens_"].tolist()
            dictionary = corpora.Dictionary(tokenized_docs)
            corpus = [dictionary.doc2bow(text) for text in tokenized_docs]

            topics = []
            for t in sorted(group[topic_col].unique()):
                # Seleciona docs pertencentes ao tópico t
                idxs = group.index[group[topic_col] == t].tolist()
                if len(idxs) == 0:
                    continue  # nenhum documento neste tópico

                words_in_topic = [tokenized_docs[group.index.get_loc(i)] for i in idxs if group.index.get_loc(i) < len(tokenized_docs)]
                flat_words = [w for sub in words_in_topic for w in sub]

                if len(flat_words) == 0:
                    continue

                # Palavras mais frequentes do tópico
                top_words = pd.Series(flat_words).value_counts().head(min(topn, len(flat_words))).index.tolist()
                if len(top_words) > 0:
                    topics.append(top_words)

            if len(topics) == 0:
                print(f"[AVISO] Nenhum tópico válido em {method} | rodada {rnd}. Pulando.")
                continue

            # Calcula coerência
            for metric in metrics:
                cm = CoherenceModel(
                    topics=topics,
                    texts=tokenized_docs,
                    dictionary=dictionary,
                    coherence=metric
                )
                coherence_score = cm.get_coherence()

                results.append({
                    "method": method,
                    "round": rnd,
                    "metric": metric.upper(),
                    "coherence": coherence_score,
                    "n_topics_valid": len(topics),
                    "n_docs": len(group),
                    "topn": topn
                })

        except Exception as e:
            print(f"[ERRO] {method} | rodada {rnd}: {e}")

    df_coherence = pd.DataFrame(results)
    return df_coherence
    
# avaliando
# df_coherence = evaluate_topic_coherence(df_topics)

df_coherence_list = []
for k in tqdm(df_all_topics.test_k.unique(), desc="Processando coherence..."):
    df_coherence = evaluate_topic_coherence(df_all_topics[df_all_topics["test_k"] == k])
    df_coherence_list.append(df_coherence)

df_coherence_ =  pd.concat(df_coherence_list, ignore_index=True)
del df_coherence_list

df_coherence_.to_parquet("../results/df_coherence.parquet")

