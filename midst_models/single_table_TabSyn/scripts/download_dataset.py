import os
import zipfile
from urllib import request


def unzip_file(zip_filepath, dest_path):
    with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
        zip_ref.extractall(dest_path)


def download_from_uci(name, data_dir, name_url_dict_uci):
    print(f"Start processing dataset {name} from UCI.")
    save_dir = f"{data_dir}/{name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

        url = name_url_dict_uci[name]
        request.urlretrieve(url, f"{save_dir}/{name}.zip")
        print(
            f"Finish downloading dataset from {url}, data has been saved to {save_dir}."
        )

        unzip_file(f"{save_dir}/{name}.zip", save_dir)
        print(f"Finish unzipping {name}.")

    else:
        print("Already downloaded.")


if __name__ == "__main__":
    DATA_DIR = "/projects/aieng/diffusion_bootcamp/data/tabular/raw_data"

    NAME_URL_DICT_UCI = {
        "adult": "https://archive.ics.uci.edu/static/public/2/adult.zip",
        "default": "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
        "magic": "https://archive.ics.uci.edu/static/public/159/magic+gamma+telescope.zip",
        "shoppers": "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip",
        "beijing": "https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip",
        "news": "https://archive.ics.uci.edu/static/public/332/online+news+popularity.zip",
    }

    for name in NAME_URL_DICT_UCI.keys():
        download_from_uci(name, DATA_DIR, NAME_URL_DICT_UCI)
