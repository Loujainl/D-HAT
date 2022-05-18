# D-HAT
### Deduplication over Heterogeneous Attribute Types

D-HAT is a learning-free pipeline for end-to-end deduplication.
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
