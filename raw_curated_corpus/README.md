The GLiNER-BioMed curated corpora are available on Hugging Face:

* [Curated corpus](https://huggingface.co/datasets/anthonyyazdaniml/gliner-biomed-curated-corpus)
* [Balanced curated corpus](https://huggingface.co/datasets/anthonyyazdaniml/gliner-biomed-balanced-curated-corpus) (equal representation across all data sources)

We also provide preprocessing code to replicate the curation steps:

```bash
# Pubmed abstracts
python A0_download_abstracts.py
python A2_abstract_cleaning_main.py --input_csv ./data/pubmed_raw_abstracts.csv --output_csv ./data/pubmed_clean_abstracts.csv
python F0_deduplicate.py --input_csv ./data/pubmed_clean_abstracts.csv --output_csv ./data/pubmed_clean_ded_abstracts.csv

# CT details
python B1_ct_detailed_main.py --input_dir ./data/clinicaltrials_gov/ctg-studies.json --output_file ./data/ct_descriptions_raw_text.csv
python B3_ct_detailed_cleaning_main.py --input_csv ./data/ct_descriptions_raw_text.csv --output_csv ./data/ct_descriptions_clean_text.csv
python F0_deduplicate.py --input_csv ./data/ct_descriptions_clean_text.csv --output_csv ./data/ct_descriptions_clean_ded_text.csv

# CT groups
python C1_ct_armgroups_main.py --input_dir ./data/clinicaltrials_gov/ctg-studies.json --output_file ./data/ct_groups_raw_text.csv
python C3_clean_armgroups_main.py --input_csv ./data/ct_groups_raw_text.csv --output_csv ./data/ct_groups_clean_text.csv
python F0_deduplicate.py --input_csv ./data/ct_groups_clean_text.csv --output_csv ./data/ct_groups_clean_ded_text.csv

# Drug labels
python D0_extract_setids.py
python D1_fetch_html_content.py
python D3_hpl_html_main.py
python D5_hpl_cleaning_main.py --input_csv ./data/hpl_html_raw_text.csv --output_csv ./data/hpl_html_clean_text.csv
python F0_deduplicate.py --input_csv ./data/hpl_html_clean_text.csv --output_csv ./data/hpl_html_clean_ded_text.csv

# Bio patents
python E1_patents_cleaning_main.py --input_csv ./data/patents-8-12.csv --output_csv ./data/patents_clean_text.csv
python F0_deduplicate.py --input_csv ./data/patents_clean_text.csv --output_csv ./data/patents_clean_ded_text.csv

# Merge data
python G0_merge_data.py
```
