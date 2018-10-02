import requests
import progressbar

from os import path, makedirs

SIMPLE_BAR = ['Progress ', progressbar.Bar(), ' ', progressbar.Percentage(), ' completed']

def maybe_download(archive_name, target_dir, archive_url):
    # If archive file does not exist, download it...
    archive_path = path.join(target_dir, archive_name)

    if not path.exists(target_dir):
        print('No path "%s" - creating ...' % target_dir)
        makedirs(target_dir)

    if not path.exists(archive_path):
        print('No archive "%s" - downloading...' % archive_path)
        req = requests.get(archive_url, stream=True)
        total_size = int(req.headers.get('content-length', 0))
        done = 0
        with open(archive_path, 'wb') as f:
            bar = progressbar.ProgressBar(max_value=total_size, widgets=SIMPLE_BAR)
            for data in req.iter_content(1024*1024):
                done += len(data)
                f.write(data)
                bar.update(done)
    else:
        print('Found archive "%s" - not downloading.' % archive_path)
    return archive_path
