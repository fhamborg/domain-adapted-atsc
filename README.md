strongly based on: https://github.com/deepopinion/domain-adapted-atsc

### Installation
First clone repository, open a terminal and cd to the repository

```
conda create --yes -n ada python=3.7
source activate ada
pip install -r requirements.txt 
conda install --yes -c anaconda scipy
conda install --yes scikit-learn
conda install --yes pytorch torchvision cudatoolkit=10.1 -c pytorch
python -m spacy download en_core_web_sm
pip install torch-transformers
mkdir -p data/raw/semeval2014  # creates directories for data
mkdir -p data/transformed
mkdir -p data/models
```


### run the code
```
# check number of non-zero lines
cat data/transformed/copewe10m.txt | sed '/^\s*$/d' | wc -l
# should be roughly 10M: 10000002

cd finetuning_and_classification/

# change to env (if not yet done)
module load anaconda
module load cuda
source activate ada

# open a screen session, because this will take a while
screen -S lmfine

# prepare the finetuning corpus
python pregenerate_training_data.py --train_corpus ../data/transformed/copewe10m.txt --bert_model bert-base-uncased --do_lower_case --output_dir copewe10m_prepared/ --epochs_to_generate 3 --max_seq_len 256

# run the finetuning
python finetune_on_pregenerated.py --pregenerated_data copewe10m_prepared --bert_model bert-base-uncased --do_lower_case --output_dir copewe10m_finetuned/ --epochs 3 --train_batch_size 16
```