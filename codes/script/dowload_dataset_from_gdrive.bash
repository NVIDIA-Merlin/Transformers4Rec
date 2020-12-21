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
      
gdown https://drive.google.com/uc?id=1YtNyC9kxM9KDxP74DTagPTLFZs7k9FuH
unzip_d ecommerce_preproc_v5.zip

cd ${cur_path}