```
python create_cadec_splits.py
python create_n2c2_splits.py
python create_tac_splits.py
python create_ncbi_splits.py
python create_bc5cdr_splits.py
python create_chia_splits.py
python create_biored_splits.py
python create_biomed_ner_splits.py

python chunk_jsons.py --input_folder ./data/cadec --output_folder ./data/cadec_chunked --max_length 512
python chunk_jsons.py --input_folder ./data/n2c2 --output_folder ./data/n2c2_chunked --max_length 512
python chunk_jsons.py --input_folder ./data/smm4h23 --output_folder ./data/smm4h23_chunked --max_length 512
python chunk_jsons.py --input_folder ./data/tac --output_folder ./data/tac_chunked --max_length 512
python chunk_jsons.py --input_folder ./data/ncbi_disease --output_folder ./data/ncbi_disease_chunked --max_length 512
python chunk_jsons.py --input_folder ./data/bc5cdr --output_folder ./data/bc5cdr_chunked --max_length 512
python chunk_jsons.py --input_folder ./data/chia --output_folder ./data/chia_chunked --max_length 512
python chunk_jsons.py --input_folder ./data/biored --output_folder ./data/biored_chunked --max_length 512
python chunk_jsons.py --input_folder ./data/biomed_ner --output_folder ./data/biomed_ner_chunked --max_length 512
```
