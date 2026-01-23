import torch.utils.data as data
import os
import os.path
import errno

class Omniglot(data.Dataset):
    urls = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    ]
    raw_folder = 'raw'  # 定义原始数据存放目录
    processed_folder = 'processed'  # 定义处理后数据存放目录
    training_file = 'training.pt'
    test_file = 'test.pt'

    '''
    The items are (filename,category). The index of all the categories can be found in self.idx_classes
    Args:
    - root: the directory where the dataset will be stored
    - transform: how to transform the input
    - target_transform: how to transform the target
    - download: need to download the dataset
    '''

    def __init__(self, root, transform=None, target_transform=None, download=False):
        self.root = root  # 数据集存储路径
        self.transform = transform  # 对输入图像的变换操作
        self.target_transform = target_transform  # 对标签的变换操作

        if not self._check_exists():
            if download:
                self.download()
            else:
                raise RuntimeError('Dataset not found.' + ' You can use download=True to download it')

        self.all_items = find_classes(os.path.join(self.root, self.processed_folder))
        self.idx_classes = index_classes(self.all_items)

    def __getitem__(self, index):
        filename = self.all_items[index][0]
        img = str.join('/', [self.all_items[index][2], filename])

        target = self.idx_classes[self.all_items[index][1]]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.all_items)  # 数据集总长度，即图像总数

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, "images_evaluation")) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, "images_background"))

    def download(self):
        import urllib.request  # 使用Python 3自带的urllib
        import zipfile

        if self._check_exists():
            return

        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        raw_path = os.path.join(self.root, self.raw_folder)
        processed_path = os.path.join(self.root, self.processed_folder)
        local_bg = os.path.join(raw_path, 'images_background.zip')
        local_eval = os.path.join(raw_path, 'images_evaluation.zip')
        if os.path.exists(local_bg) and os.path.exists(local_eval):
            for file_path in [local_bg, local_eval]:
                print("== Unzip from " + file_path + " to " + processed_path)
                zip_ref = zipfile.ZipFile(file_path, 'r')  # 解压
                zip_ref.extractall(processed_path)
                zip_ref.close()
            print("Local dataset found. Extraction finished.")
            return

        for url in self.urls:
            print('== Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(raw_path, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            print("== Unzip from " + file_path + " to " + processed_path)
            zip_ref = zipfile.ZipFile(file_path, 'r')  # 解压
            zip_ref.extractall(processed_path)
            zip_ref.close()
        print("Download finished.")

# 获取所有数据项
def find_classes(root_dir):
    retour = []
    for (root, dirs, files) in os.walk(root_dir):
        for f in files:
            if (f.endswith("png")):  # 只处理以".png"结尾的图像文件
                r = root.split('/')
                lr = len(r)
                retour.append((f, r[lr - 2] + "/" + r[lr - 1], root))
    print("== Found %d items " % len(retour))
    return retour

# 建立类别到索引的映射
def index_classes(items):
    idx = {}
    for i in items:
        if i[1] not in idx:
            idx[i[1]] = len(idx)  # 每个唯一类别都被分配了一个连续的整数索引
    print("== Found %d classes" % len(idx))
    return idx