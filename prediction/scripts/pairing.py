import pandas as pd
import numpy as np
import json
from pathlib import Path


def load_config(path=None):

    if path is None:

        # pairing.py 所在資料夾
        base_dir = Path(__file__).resolve().parent

        path = base_dir / "config.json"

    with open(path) as f:
        config = json.load(f)

    gene_to_idx = config["gene_to_idx"]

    STATE_PROP = config["state_prop"]

    NUM_GENES = config["num_genes"]

    return gene_to_idx, STATE_PROP, NUM_GENES


gene_to_idx, STATE_PROP, NUM_GENES = load_config()

def encode_condition(pert):

    genes = pert.split("+")

    vec = np.zeros(NUM_GENES)

    for g in genes:
        vec[gene_to_idx[g]] += 1

    return vec


def sample_nc_cells(
    adata_nc,
    batches,
    batch_target,
    state_col="state",
    batch_col="SampleID"
):

    selected_cells = []

    for batch in batches:

        batch_cells = adata_nc[
            adata_nc.obs[batch_col] == batch
        ]

        n_batch = batch_target[batch]

        chosen = []

        # ensure 1 lipo_adipo
        lipo_cells = batch_cells[
            batch_cells.obs[state_col] == "lipo_adipo"
        ]

        if len(lipo_cells) > 0:

            chosen_lipo = np.random.choice(
                lipo_cells.obs_names,
                size=1,
                replace=False
            )

            chosen.extend(chosen_lipo)

        remaining_n = n_batch - len(chosen)

        pool = batch_cells[
            ~batch_cells.obs_names.isin(chosen)
        ]

        state_counts = {

            s: int(round(STATE_PROP[s] * remaining_n))
            for s in STATE_PROP
            if s != "lipo_adipo"
        }

        diff = remaining_n - sum(state_counts.values())

        if diff > 0:

            ordered_states = sorted(
                state_counts,
                key=lambda s: STATE_PROP[s],
                reverse=True
            )

            for i in range(diff):

                state_counts[
                    ordered_states[i % len(ordered_states)]
                ] += 1

        for state, n_pick in state_counts.items():

            state_cells = pool[
                pool.obs[state_col] == state
            ]

            if len(state_cells) == 0:
                continue

            n_pick = min(n_pick, len(state_cells))

            sampled = np.random.choice(
                state_cells.obs_names,
                size=n_pick,
                replace=False
            )

            chosen.extend(sampled)

        remaining_pool = batch_cells[
            ~batch_cells.obs_names.isin(chosen)
        ]

        if len(chosen) < n_batch:

            extra = np.random.choice(
                remaining_pool.obs_names,
                size=n_batch - len(chosen),
                replace=False
            )

            chosen.extend(extra)

        selected_cells.extend(chosen)

    return selected_cells

def compute_batch_targets(batches, total_cells):
    
    base = total_cells // len(batches)

    batch_target = {b: base for b in batches}

    remainder = total_cells - base * len(batches)

    extra_batches = np.random.choice(
        batches,
        size=remainder,
        replace=False
    )

    for b in extra_batches:
        batch_target[b] += 1

    return batch_target


def sample_diverse_nc(
    adata_nc,
    n_target,
    random_state=0,
    min_cells=4,
    state_col="state",
    batch_col="SampleID"
):

    rng = np.random.default_rng(random_state)

    n_target = max(n_target, min_cells)

    df = adata_nc.obs.copy()

    selected = []

    states = [
        "lipo_adipo",
        "lipo",
        "adipo",
        "pre_adipo",
        "other"
    ]

    for s in states:

        pool = df[df[state_col] == s]

        if len(pool) == 0:
            continue

        chosen = rng.choice(
            pool.index,
            size=1,
            replace=False
        )

        selected.extend(chosen)

        if len(selected) >= n_target:
            break

    remaining_n = n_target - len(selected)

    if remaining_n > 0:

        pool = df.loc[
            ~df.index.isin(selected)
        ]

        batches = pool[batch_col].unique()

        batch_targets = compute_batch_targets(
            batches,
            remaining_n
        )

        for b in batches:

            batch_pool = pool[
                pool[batch_col] == b
            ]

            n_pick = batch_targets[b]

            if len(batch_pool) == 0:
                continue

            n_pick = min(
                n_pick,
                len(batch_pool)
            )

            chosen = rng.choice(
                batch_pool.index,
                size=n_pick,
                replace=False
            )

            selected.extend(chosen)

    return adata_nc[selected]


def assign_state_label(df):

    labels = []

    for _, r in df.iterrows():

        if (r["lipo"] == 1) and (r["adipo"] == 1):
            labels.append("lipo_adipo")

        elif r["lipo"] == 1:
            labels.append("lipo")

        elif r["adipo"] == 1:
            labels.append("adipo")

        elif r["pre_adipo"] == 1:
            labels.append("pre_adipo")

        else:
            labels.append("other")

    return pd.Series(labels, index=df.index)

def format_state_prop(
    df,
    state_col="state"
):

    prop = (
        df[state_col]
        .value_counts(normalize=True)
    )

    keys = [
        "adipo",
        "lipo_adipo",
        "other",
        "pre_adipo"
    ]

    return ", ".join(

        f"{k} {prop.get(k,0):.3f}"

        for k in keys
    )


import re


def simplify_batch_name(batch_name):

    match = re.search(r'_(\d+)$', batch_name)

    if match:
        return f"P{match.group(1)}"

    return batch_name



def format_batch_prop(
    df,
    batch_col="SampleID",
    max_show=6
):

    prop = (
        df[batch_col]
        .value_counts(normalize=True)
    )

    items = []

    for k, v in prop.items():

        short_k = simplify_batch_name(k)

        items.append(
            f"{short_k} {v:.3f}"
        )

    if len(items) > max_show:

        items = items[:max_show] + ["..."]

    return ", ".join(items)

import numpy as np


def get_primary_control(pert):
    
    n_guides = len(pert.split("+"))

    if n_guides == 1:
        return "NC"

    # 2 guides or more → 使用 NC+NC
    return "NC+NC"


def make_all_pairs(
    ctrl_latent,
    pert_latent,
    cond_vec
):

    pairs = []

    for i in range(len(ctrl_latent)):

        for j in range(len(pert_latent)):

            pairs.append({

                "z0": ctrl_latent[i],
                "z1": pert_latent[j],
                "cond": cond_vec

            })

    return pairs
def make_random_pairs(
    ctrl_latent,
    pert_latent,
    cond_vec,
    rng
):

    n = min(
        len(ctrl_latent),
        len(pert_latent)
    )

    ctrl_sample = rng.choice(
        ctrl_latent,
        size=n,
        replace=len(ctrl_latent) < n
    )

    pairs = []

    for i in range(n):

        pairs.append({

            "z0": ctrl_sample[i],
            "z1": pert_latent[i],
            "cond": cond_vec

        })

    return pairs

def is_single_gene_with_nc(pert):
    
    genes = pert.split("+")

    non_nc = [

        g for g in genes

        if g != "NC"

    ]

    return (

        len(non_nc) == 1

        and len(genes) > 1

    )
    
    
import pickle
import numpy as np
from pathlib import Path


AVAILABLE_GENES = [
    'CEBPA','CEBPB','CEBPD','CREB1','FOXO1','KIF11','KLF15',
    'MLXIPL','NC','NR3C1','POLR2D','PPARG','PPARG2','SF3B1',
    'SREBF1','STAT5A','STAT5B','TCF7L2','ZBED3'
]


def load_gene_embeddings(
    size="large",
    resource_dir="../../resources"
):
    """
    Load GPT gene embeddings.

    Parameters
    ----------
    size : str
        "large" or "small"

    resource_dir : str
        folder containing pickle file

    Returns
    -------
    dict
        gene -> embedding vector
    """

    fname = {
        "large": "GPT_latest_gene_large_embeddings.pickle",
        "small": "GPT_latest_gene_small_embeddings.pickle"
    }[size]

    path = Path(resource_dir) / fname

    with open(path, "rb") as f:
        emb = pickle.load(f)

    return emb



def encode_condition_embedding(
    condition,
    embedding_dict,
    add_num_guides=True
):
    """
    Convert perturbation string to embedding vector
    using precomputed perturbation embeddings.

    Example
    -------
    "PPARG+NC"
    "PPARG+NC+NC"
    "STAT5A+STAT5B"

    Parameters
    ----------
    condition : str

    embedding_dict : dict
        mapping:
        perturbation -> embedding vector

    add_num_guides : bool
        append number of guide RNAs

    Returns
    -------
    np.ndarray
    """

    if condition not in embedding_dict:

        raise ValueError(
            f"{condition} not found in embedding_dict"
        )

    vec = embedding_dict[condition]

    if add_num_guides:

        n_guides = len(condition.split("+"))

        vec = np.concatenate([
            vec,
            np.array([n_guides])
        ])

    return vec

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import numpy as np

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def make_ot_pairs(

    ctrl_latent,
    pert_latent,
    cond_vec,

    max_distance=8,
):

    cost = cdist(ctrl_latent, pert_latent, metric="euclidean")

    cost[cost > max_distance] = 1e6

    row_ind, col_ind = linear_sum_assignment(cost)

    pairs = []

    for i,j in zip(row_ind, col_ind):

        if cost[i,j] < max_distance:

            pairs.append({

                "z0": ctrl_latent[i],

                "z1": pert_latent[j],

                "cond": cond_vec

            })

    return pairs

       

def make_ot_pairs_bootstrap(

    ctrl_latent,
    pert_latent,
    cond_vec,
    n_repeat=3,
    rng=None
):

    if rng is None:

        rng = np.random.default_rng()

    pairs = []

    for _ in range(n_repeat):

        ctrl_sample = ctrl_latent[

            rng.choice(

                len(ctrl_latent),

                size=len(ctrl_latent),

                replace=True
            )

        ]

        new_pairs = make_ot_pairs(

            ctrl_sample,

            pert_latent,

            cond_vec
        )

        pairs.extend(new_pairs)

    return pairs
    
    
def get_condition_encoder(
    condition_mode="onehot",
    embedding_dict=None,
    add_num_guides=True
):

    if condition_mode == "onehot":

        return lambda pert: encode_condition(pert)


    elif condition_mode == "embedding":

        if embedding_dict is None:

            raise ValueError(
                "embedding_dict required"
            )


        return lambda pert: encode_condition_embedding(

            pert,

            embedding_dict,

            add_num_guides=add_num_guides
        )


    else:

        raise ValueError(
            "condition_mode must be one of:\n"
            "onehot\n"
            "embedding"
        )
        
def build_pairs_from_anndata(
    adata,
    latent_key="X_pca",
    state_col="state",
    batch_col="SampleID",
    gene_col="gene",

    # condition encoding
    condition_mode="onehot",
    embedding_dict=None,
    add_num_guides=True,
    normalize_embedding=False,

    small_threshold=200,
    random_state=0,
    verbose=False,
    n_repeat_ot_pairs = 3
):

    rng = np.random.default_rng(random_state)

    latent = adata.obsm[latent_key]

    obs_gene = adata.obs[gene_col].values

    unique_perts = np.unique(obs_gene)

    pert_to_idx = {

        p: np.where(obs_gene == p)[0]

        for p in unique_perts
    }

    pairs = []

    # choose encoder
    encode_fn = get_condition_encoder(

        condition_mode=condition_mode,

        embedding_dict=embedding_dict,

        add_num_guides=add_num_guides,

        normalize=normalize_embedding
    )


    for pert in unique_perts:

        if pert in ["NC","NC+NC"]:
            continue


        cond_vec = encode_fn(pert)

        pert_idx = pert_to_idx[pert]

        pert_latent = latent[pert_idx]


        primary_control = get_primary_control(pert)

        ctrl_idx = pert_to_idx[primary_control]

        ctrl_latent = latent[ctrl_idx]


        if min(

            len(ctrl_latent),

            len(pert_latent)

        ) < small_threshold:

            new_pairs = make_all_pairs(

                ctrl_latent,

                pert_latent,

                cond_vec

            )

            strategy = "all-pairs"

        else:

            new_pairs = make_ot_pairs_bootstrap(

                ctrl_latent,

                pert_latent,

                cond_vec,
                n_repeat=n_repeat_ot_pairs,

            )

            strategy = "ot_pairs"


        pairs.extend(new_pairs)


        print(

            f"{pert} {len(new_pairs)} {strategy}"

        )


        if verbose:

            target_df = adata.obs.iloc[pert_idx]

            ctrl_df = adata.obs.iloc[ctrl_idx]


            print(f"\n{pert} distribution")


            print(

                f"target state: {format_state_prop(target_df,state_col)}"

            )


            print(

                f"{primary_control} state: {format_state_prop(ctrl_df,state_col)}"

            )


            print()


            print(

                f"target batch: {format_batch_prop(target_df,batch_col)}"

            )


            print(

                f"{primary_control} batch: {format_batch_prop(ctrl_df,batch_col)}"

            )


            print()



        # fallback NC
        if is_single_gene_with_nc(pert):

            adata_nc = adata[

                adata.obs[gene_col] == "NC"

            ]


            adata_nc_sampled = sample_diverse_nc(

                adata_nc,

                len(pert_latent),

                random_state,

                state_col=state_col,

                batch_col=batch_col

            )


            nc_latent = adata_nc_sampled.obsm[latent_key]

            extra_pairs = make_ot_pairs_bootstrap(

                nc_latent,

                pert_latent,

                cond_vec,
                n_repeat=n_repeat_ot_pairs,


            )


            pairs.extend(extra_pairs)


            print(

                f"{pert} {len(extra_pairs)} random (NC fallback)"

            )


    return pairs