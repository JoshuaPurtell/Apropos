import glob
import os

from apropos.src.core.lms.api_caching_fxns import SafeCache, ThreadedCache

cache = SafeCache("apropos/src/lms/.cache", "apropos/src/lms/.cache.db")
threaded_cache = ThreadedCache("apropos/src/lms/.cache")


def print_cache_folder_sizes():
    cache_folders = glob.glob("apropos/src/lms/.cache")
    db_files = glob.glob("apropos/src/lms/.*.db")

    for db_file in db_files:
        if os.path.exists(db_file):
            db_size = os.path.getsize(db_file)
            print(f"{db_file} (SQLite): {db_size / (1024 * 1024):.2f} MB")

    for folder in cache_folders:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(folder):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        print(f"{folder}: {total_size / (1024 * 1024):.2f} MB")


# print_cache_folder_sizes()
