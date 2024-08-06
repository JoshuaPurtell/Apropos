from src.lms.api_caching_fxns import SafeCache
import os
import glob

cache = SafeCache("src/lms/.cache", "src/lms/.cache.db")

def print_cache_folder_sizes():
    cache_folders = glob.glob("src/lms/.cache")
    db_files = glob.glob("src/lms/.*.db")
    
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

print_cache_folder_sizes()