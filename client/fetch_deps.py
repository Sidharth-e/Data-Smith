import urllib.request
import tarfile
import os
import shutil

pkgs = [
    ("axios", "https://registry.npmjs.org/axios/-/axios-1.7.9.tgz", "node_modules/axios"),
    ("zustand", "https://registry.npmjs.org/zustand/-/zustand-5.0.3.tgz", "node_modules/zustand"),
    ("lucide-react", "https://registry.npmjs.org/lucide-react/-/lucide-react-0.475.0.tgz", "node_modules/lucide-react"),
    ("react-query", "https://registry.npmjs.org/@tanstack/react-query/-/react-query-5.66.0.tgz", "node_modules/@tanstack/react-query"),
]

for name, url, dest in pkgs:
    tar_path = f"{name}.tgz"
    print(f"Downloading {name} from {url}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp, open(tar_path, 'wb') as out_file:
        shutil.copyfileobj(resp, out_file)
    
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(tar_path, 'r:gz') as tar:
        for member in tar.getmembers():
            # strip leading 'package/' directory from tar
            if member.name.startswith('package/'):
                member.name = member.name[len('package/'):]
            if member.name:
                tar.extract(member, path=dest)
    
    if os.path.exists(tar_path):
        os.remove(tar_path)
    print(f"Extracted {name} into {dest}")

print("All dependencies fetched and extracted successfully!")
