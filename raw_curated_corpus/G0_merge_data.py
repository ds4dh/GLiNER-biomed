import pandas as pd

def main():
    # Load datasets
    HPL = pd.read_csv("./data/hpl_html_clean_ded_text.csv")
    CTD = pd.read_csv("./data/ct_descriptions_clean_ded_text.csv")
    CTG = pd.read_csv("./data/ct_groups_clean_ded_text.csv")
    PA = pd.read_csv("./data/pubmed_clean_ded_abstracts.csv")
    PAT = pd.read_csv("./data/patents_clean_ded_text.csv")
    
    # Add source column to identify datasets
    HPL["source"] = "HPL"
    CTD["source"] = "CTD"
    CTG["source"] = "CTG"
    PA["source"] = "PA"
    PAT["source"] = "PAT"
    
    # Ensure 'source' is the first column
    HPL = HPL[["source"] + [col for col in HPL.columns if col != "source"]]
    CTD = CTD[["source"] + [col for col in CTD.columns if col != "source"]]
    CTG = CTG[["source"] + [col for col in CTG.columns if col != "source"]]
    PA = PA[["source"] + [col for col in PA.columns if col != "source"]]
    PAT = PAT[["source"] + [col for col in PAT.columns if col != "source"]]
    
    # Merge all datasets
    merged_all = pd.concat([HPL, CTD, CTG, PA, PAT], ignore_index=True)
    
    # Balance the datasets
    min_size = min(len(HPL), len(CTD), len(CTG), len(PA), len(PAT))
    seed = 42
    
    HPL_sampled = HPL.sample(n=min_size, random_state=seed)
    CTD_sampled = CTD.sample(n=min_size, random_state=seed)
    CTG_sampled = CTG.sample(n=min_size, random_state=seed)
    PA_sampled = PA.sample(n=min_size, random_state=seed)
    PAT_sampled = PAT.sample(n=min_size, random_state=seed)
    
    balanced_merged = pd.concat([HPL_sampled, CTD_sampled, CTG_sampled, PA_sampled, PAT_sampled], ignore_index=True)
    
    # Shuffle the merged datasets
    merged_all = merged_all.sample(frac=1, random_state=seed).reset_index(drop=True)
    balanced_merged = balanced_merged.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Save outputs to CSV
    merged_all.to_csv("./data/bio_text_ay.csv", index=False)
    balanced_merged.to_csv("./data/balanced_bio_text_ay.csv", index=False)

if __name__ == "__main__":
    main()