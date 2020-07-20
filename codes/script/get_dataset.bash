pip install gdown

unzip_d () {
    zipfile="$1"
    zipdir=${1%.zip}
    unzip -d "$zipdir" "$zipfile"
}

cur_path=$(pwd)

target_path="/root/dataset"
mkdir -p ${target_path}

cd ${target_path}

gdown https://drive.google.com/uc?id=1f0PV3CLyN1EYzmLXBDeprqST-rf1ZaY3
unzip ecommerce_preproc_neg_samples_50_strategy_cooccurrence_19_days_first_10k_sessions.parquet.zip

gdown https://drive.google.com/uc?id=1AfyZ75KH7XwOWHZ7qY0CDix9yxIn1ejL
unzip_d ecommerce_preproc_neg_samples_50_strategy_uniform-2019-10.zip

gdown https://drive.google.com/uc?id=1Z5e4qjRNSM36zaOAJ1zWV9WXZZVGR2sh
unzip_d ecommerce_preproc_neg_samples_50_strategy_recent_popularity-2019-10.zip

cd ${cur_path}