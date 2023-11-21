[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deduplication-over-heterogeneous-attribute/entity-resolution-on-amazon-google)](https://paperswithcode.com/sota/entity-resolution-on-amazon-google?p=deduplication-over-heterogeneous-attribute)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deduplication-over-heterogeneous-attribute/entity-resolution-on-abt-buy)](https://paperswithcode.com/sota/entity-resolution-on-abt-buy?p=deduplication-over-heterogeneous-attribute)

# D-HAT
### Deduplication over Heterogeneous Attribute Types

D-HAT is an end-to-end unsupervised entity matching system to link two data sources or deduplicate one.
D-HAT goes beyond existing works in three ways:
1. Deduplicates data sets with *heterogeneous* types of attributes and missing values.
2. Inherently supports and leverages complex schemata of *high dimensionality*.
3. Achieves state-of-the-art results in a short run time without requiring any labelled data.

- To install requirements: `pip install -r /path/to/requirements.txt`

- Execute EmbedOrigDHAT with the pathtofile+filename `python3 EmbedOrigDHAT.py pathtosource pathtotarget`

   For example: `python3 EmbedOrigDHAT.py Abt_Buy/1_abt Abt_Buy/2_buy`
   * The user will be prompted to specify:
    1. The run mode [local, cluster]
    2. Name of truth JSON file `abtbuy`
    3. Configuration for D-HAT: 
    * Syntactic, Semantic, or Hybrid
    * Dirty or Clean-Clean Task

### Citation
If you use D-HAT, please cite the following paper:
<br/>
_Liekah, L., Papadakis, G. (2022). Deduplication Over Heterogeneous Attribute Types (D-HAT). In: Chen, W., Yao, L., Cai, T., Pan, S., Shen, T., Li, X. (eds) Advanced Data Mining and Applications. ADMA 2022. Lecture Notes in Computer Science(), vol 13726. Springer._
