import os
import urllib.request
import time

def download_tinystories():
    url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
    dest_dir = "data"
    dest_file = os.path.join(dest_dir, "tinystories.txt")
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created directory: {dest_dir}")
        
    if os.path.exists(dest_file):
        print(f"Dataset already exists at {dest_file}")
        return
        
    print(f"Downloading TinyStories dataset from {url}...")
    start_time = time.time()
    
    try:
        def report_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded / total_size) * 100)
                print(f"\rDownloading: {percent:.1f}% ({downloaded / (1024*1024):.2f} MB of {total_size / (1024*1024):.2f} MB)", end="")
            else:
                print(f"\rDownloaded {downloaded / (1024*1024):.2f} MB", end="")
        
        urllib.request.urlretrieve(url, dest_file, reporthook=report_hook)
        print(f"\nDownload finished in {time.time() - start_time:.2f} seconds.")
        print(f"Saved to: {dest_file}")
    except Exception as e:
        print(f"\nError downloading dataset: {e}")

if __name__ == "__main__":
    download_tinystories()
