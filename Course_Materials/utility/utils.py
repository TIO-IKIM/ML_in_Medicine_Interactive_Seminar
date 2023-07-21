import torch, torchvision
import PIL
import numpy as np
from tqdm.auto import tqdm
import os
from typing import List, Iterable, Callable

def tqdm_wrapper(it: Iterable, w: bool, total = None):
    if w:
        if total is None:
            return tqdm(it)
        else:
            return tqdm(it, total = total)
    else:
        return it

def getFileList(path: str, fileType: str = '.png', layer: int = 0, pbar = True):
    """
    Compiles a list of path/to/files by recursively checking the input path a list of all directories
    and sub-directories for existence of a specific string at the end.
    
    Inputs:
    
    path - Path-like or string, must point to the directory you want to check for files.
    fileType - String, all files that end with this string are selected.
    
    Outputs:
    
    fileList - A list of all paths/to/files
    """
    
    fileList = []

    # Return empty list if path is a file
    if os.path.isfile(path) and path.endswith(fileType):
        return [path]

    if os.path.isdir(path):
        if layer == 0:
            for d in tqdm_wrapper(sorted(os.listdir(path)), w=pbar):
                new_path = os.path.join(path, d)
                fileList += getFileList(new_path, fileType, layer=layer+1)
        else:
            for d in sorted(os.listdir(path)):
                new_path = os.path.join(path, d)
                fileList += getFileList(new_path, fileType, layer=layer+1)
    else:
        # Should never be called
        return []

    return sorted(fileList)

def shallow_getFileList(path: str, fileType: str = ".png", pbar: bool = True):
    fileList = [os.path.join(path, file) for file in tqdm_wrapper(sorted(os.listdir(path)), w = pbar) if str(file).endswith(fileType)]
    return fileList

def accuracy(predictions: torch.Tensor, targets: torch.Tensor):
    predicted_classes = torch.argmax(predictions, dim = 1)
    hits = [1 if predicted == target else 0 for predicted, target in zip(predicted_classes, targets)]
    acc = sum(hits)/len(targets)
    return acc

def convert_tensor_to_opencv_array(tensor: torch.Tensor, as_type = None):
    """
    Convert a tensor from CHW tensor to HWC numpy array. Number of leading dimensions can be arbitrary.
    If the tensor is already in HWC format, no operation is performed except casting to the array with as_type.
    If the tensor has no channel dimension, a dimension will be added at dim=-1.
    If it is unclear what the channel dimension is, raise an error.
    """
    s = tensor.size()
    if len(s) == 2:
        tensor = tensor.unsqueeze(-1)
        s = tensor.size()
    if s[-1] not in [1, 3]: # Is not HWC
        if s[-3] in [1, 3]: # Is CHW
            array = tensor.movedim([-2, -1, -3], [-3, -2, -1]).detach().numpy()
        else: # Is unknown format
            raise ValueError(f"Tensor has unusable channel dimensions at dim=-3/-1: {s[-3]}/{s[-1]}.")
    if as_type is not None:
        array = array.astype(as_type)
    return array

def convert_opencv_array_to_tensor(array: np.ndarray, as_type = None):
    """
    Convert an array from HWC numpy array to CHW tensor. Number of leading dimensions can be arbitrary.
    If the tensor is already in HWC format, no operation is performed except casting to the tensor with as_type.
    If it is unclear what the channel dimension is, raise an error.
    """
    tensor = torch.tensor(array)
    s = tensor.size()
    if len(s) == 2:
        tensor = tensor.unsqueeze(-1)
        s = tensor.size()
    if s[-3] not in [1, 3]: # Is not CHW
        if s[-1] in [1, 3]: # Is HWC
            tensor = torch.moveaxis(tensor, [-1, -3, -2], [-3, -2, -1])
        else: # Is unknown format
            raise ValueError(f"Tensor has unusable channel dimensions at dim=-3/-1: {s[-3]}/{s[-1]}.")
    if as_type is not None:
        tensor = tensor.to(dtype = as_type)
    return tensor

def get_num_params(model: torch.nn.Module, pbar: bool = False):
    """
    Count the named and unnamed parameters of a model.
    If the values in the tuple are not the same, something is probably quite broken in the model.
    """
    names = []
    nums = []
    np = 0
    for n, p in tqdm_wrapper(list(model.named_parameters()), w=pbar):
        names.append(n)
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        np += nn
        nums.append(nn)
    pp = 0
    for p in tqdm_wrapper(list(model.parameters()), w=pbar):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return np, pp

def unravel_model(model: torch.nn.Module, shorthand: bool = True):
    """
    Unravel a model, spitting out all named layers in instantiation (not necessarily execution) order.
    """
    params = {}
    np = 0
    for n, p in list(model.named_parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        np += nn
        params[n] = sh(nn) if shorthand is True else nn
    return params

def sh(x):
    assert isinstance(x, float) or isinstance(x, int)
    ls=["","k","M","B","T","Q"]
    i=0
    if abs(x)<1:
        return "0"
    s="-" if x<0 else ""
    while i!=(len(ls)+1):
        if i==6:
            raise ValueError(f"x is >= 1e{3*len(ls)} and can't be shorthanded. Wow.")
        elif x//(10**(3*i))>=1 and x//(10**(3*(i+1)))==0:
            break
        else:
            i+=1
    if i==0:
        return f"{s}{x}"
    return f"{s}{x/(10**(3*i)):.1f}{ls[i]}"

def lits_image_loader(file: str):

    # Open the file with PIL.
    with PIL.Image.open(file) as f:
        
        # Convert the image to grayscale
        f = f.convert("L")

        # Grab the array from the image
        image_array = np.array(f, dtype = np.float32)

    # Cram it into a tensor
    image_tensor = convert_opencv_array_to_tensor(image_array, as_type = torch.float32)

    return image_tensor

def lits_classification(file: str):

    """
    If you've made your way here - hi. I'm making the notebooks at the moment.
    I don't know how much programming everyone has done. If you have done some,
    I imagine you can already tell that reading in the csv for *every* __getitem__
    call is *terrible* programming. I know, I know.

    Look at it this way - if I wanted to do this properly, I'd classify all my
    data once, at the start, in my dataset's __init__ method and I would probably
    use pandas or a dictionary so I could pair up index, filename and target, or
    even cache the data and pair up index, image tensor and target.
    
    However, I decided to instead keep the dataset example as simple and minimalistic
    as humanly possible, to help those who have never written any code, let alone
    PyTorch code, and simply hid the ugly part.

    The result is slow.
    """

    # First, find the appropriate classes.csv, given any file, by cutting off the filename and folder.
    split = file.split("/")
    fp = "/".join(split[:-2]) # file path
    fn = split[-1] # file name

    # Open it (this is relatively slow)
    with open(f"{fp}/classes.csv", "r") as o:

        # Check line by line until finding the line we want (this is the horribly slow part)
        lines = o.readlines()
        for line in lines:
            # print(line.split(",")[0], fn)
            if line.startswith(fn):
                volume_file, liver_file, liver_visible, tumors_file, tumors_visible = line.split(",")
                if "True" in tumors_visible:
                    return torch.tensor(2, dtype = torch.long)
                elif "True" in liver_visible:
                    return torch.tensor(1, dtype = torch.long)
                else:
                    return torch.tensor(0, dtype = torch.long)

def lits_classification_loader(file: str):

    image_tensor = lits_image_loader(file = file)

    target = lits_classification(file = file)

    return image_tensor, target

def lits_segmentation_loader(file: str):

    # Open the file with PIL.
    with PIL.Image.open(file) as f:
        
        # This time, we convert the image to grayscale immediately.
        # It always was already grayscale of course, but had 3 channels - RGB.
        f = f.convert("L")

        # We grab the array from the image.
        image_array = np.array(f, dtype = np.uint8)
    
    image_tensor = convert_opencv_array_to_tensor(image_array, as_type = torch.float32)

    # Also load the segmentation masks.
    fp = "/".join(file.split("/")[:-2])
    volume_no = file.split("/")[-1].split("-")[1].split("_")[0]
    slice_no = file.split("/")[-1].split("_")[1].split(".")[0]
    with PIL.Image.open(f"{fp}/segmentations/segmentation-{volume_no}_livermask_{slice_no}.png") as f:
        f = f.convert("L")
        liver_array = np.array(f, dtype = np.uint8)
    with PIL.Image.open(f"{fp}/segmentations/segmentation-{volume_no}_lesionmask_{slice_no}.png") as f:
        f = f.convert("L")
        tumors_array = np.array(f, dtype = np.uint8)

    liver_tensor = convert_opencv_array_to_tensor(liver_array, as_type=torch.float32)
    tumors_tensor = convert_opencv_array_to_tensor(tumors_array, as_type=torch.float32)

    return image_tensor, liver_tensor, tumors_tensor

def save_model(name, model):
    loc = os.path.join("../models", name)
    os.makedirs("../models", exist_ok = True)

    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), loc)
    else:
        torch.save(model.state_dict(), loc)

def csv_logger(logfile, content, first=False, overwrite=False):
    # Make dirs if not existing yet
    os.makedirs("/".join(logfile.split("/")[:-1]), exist_ok=True)

    if os.path.exists(logfile) and overwrite is False:
        if first is True:
            print("Logfile already exists and overwrite is False. Logging to existing file.")
        with open(logfile, "a+") as c:
            c.write(','.join(str(item) for item in list(content.values()))+"\n")
    elif (os.path.exists(logfile) and overwrite is True) or not os.path.exists(logfile):
        if os.path.exists(logfile) and first is True:
            os.remove(logfile)
        with open(logfile, "a+") as c:
            if first is True:
                c.write(','.join(str(name) for name in list(content.keys()))+"\n")
            c.write(','.join(str(item) for item in list(content.values()))+"\n")
    else:
        pass

class CacheError(Exception):
    pass

class LiTS_Classification_Dataset(torch.utils.data.Dataset):

    """
    Dataset for the LiTS challenge - this one loads data with targets intended for classification.
    data_dir is the parent_folder of the dataset.
    transforms can be any callable that takes and returns a valid tensor that your network accepts.
    verbose controls the verbosity of the dataset.
    cache_data caches the data in the worker holding the dataset if set to True. Otherwise, read from disk.
    cache_mgr, if provided, should be a Manager, whose namespace contains a .data dict, which in turn has the
    "train", "val", and "test" keys. In this case, all workers cache to the memory of the Manager's process.
    """

    def __init__(
        self, 
        data_dir: str, 
        transforms: Callable = None, 
        verbose: bool = False, 
        cache_data: bool = False, 
        cache_mgr: object = None,
        debug: bool = False):

        super(LiTS_Classification_Dataset, self).__init__()
        
        self.data_dir = data_dir
        if not all([any([y for x in os.listdir(self.data_dir)]) for y in ["train", "val", "test"]]):
            raise ValueError(f"{self.data_dir} does not contain a 'train', 'val', and 'test' directory and is therefore probably wrong.")
        
        self.transforms = transforms
        if not callable(self.transforms) and not (self.transforms is None):
            raise TypeError(f"If transforms are provided, transforms must be a callable.")
        
        self.verbose = verbose
        
        self.cache_data = cache_data
        
        self.cache_mgr = cache_mgr
        if not (self.cache_mgr is None) and not all([x in list(self.cache_mgr.data.keys()) for x in ["train", "val", "test"]]):
            raise KeyError("Cache managers must be a Proxy for a dict object with the keys 'train', 'val', and 'test'.")
        if not (self.cache_mgr is None) and not all([x in list(self.cache_mgr.cached.keys()) for x in ["train", "val", "test"]]):
            raise KeyError("Cache managers must be a Proxy for a dict object with the keys 'train', 'val', and 'test'.")

        if self.cache_data is True and cache_mgr is not None:
            self.data = self.cache_mgr.data
            self.cached = self.cache_mgr.cached
        else:
            if self.cache_data is True:
                print("Warning: Caching without a cache manager leads to the cache being reset every epoch, unless num_workers is 0!")
            self.data = {
                "train": {},
                "val": {},
                "test": {}
                }
            self.cached = {
            "train": False,
            "val": False,
            "test": False
        }
        
        self.debug = debug

        self.mode = "train"

        self._make_file_names()
        self._make_targets()
        if self.verbose:
            print("Dataset initialization complete.")

    def _class_nr(self, lv: str, tv: str):
        if "True" in tv:
            return 2
        elif "True" in lv:
            return 1
        else:
            return 0

    def _make_file_names(self):
        
        train_d = os.path.join(self.data_dir, "train", "volumes")
        val_d = os.path.join(self.data_dir, "val", "volumes")
        test_d = os.path.join(self.data_dir, "test", "volumes")
        train_f = shallow_getFileList(path = train_d, fileType = ".png", pbar = self.verbose)
        val_f = shallow_getFileList(path = val_d, fileType = ".png", pbar = self.verbose)
        test_f = shallow_getFileList(path = test_d, fileType = ".png", pbar = self.verbose)
        self.file_names = {
            "train": {i: file for i, file in enumerate(train_f)},
            "val": {i: file for i, file in enumerate(val_f)},
            "test": {i: file for i, file in enumerate(test_f)}
        }

    def _make_targets(self):

        self.targets = {}
        for s in ["train", "val", "test"]:
            file = self.file_names[s][0]
            split = file.split("/")
            fp = "/".join(split[:-2])
            with open(f"{fp}/classes.csv", "r") as o:
                lines = o.readlines()
            self.targets[s] = {f"{fp}/volumes/{vf}": torch.tensor(self._class_nr(lv, tv)) for vf, lf, lv, tf, tv in [line.split(",") for line in tqdm_wrapper(lines[1:], w = self.verbose)]}
    
    def __len__(self, subset: str = None):

        if subset is None:
            return len(self.file_names[self.mode])
        else:
            return len(self.file_names[subset])

    def __getitem__(self, idx: int, layer: int = 0):

        try:
            # Find filename
            file_name = self.file_names[self.mode][idx]
            # If needed, load from disk
            if self.cache_data is False or self.cached[self.mode] is False:
                image_tensor = lits_image_loader(file = file_name)
            # If possible, try to load from cache
            elif self.cache_data is True and self.cached[self.mode] is True:
                try:
                    image_tensor = self.data[self.mode][file_name].to(dtype = torch.float32)
                except KeyError:
                    raise CacheError(f"Expected file {file_name} at index {idx} to be in cache, but got KeyError. File may be unavailable.")
            # Normalize to [0,1]
            image_tensor -= torch.min(image_tensor)
            image_tensor /= torch.max(image_tensor)
            # Save to cache, if required, cast to fp16 to save room
            if self.cache_data is True and self.cached[self.mode] is False:
                    self.data[self.mode][file_name] = image_tensor.to(torch.float16)
            # Transform training data
            if self.transforms is not None and self.mode == "train":
                image_tensor = self.transforms(image_tensor)
            # Get targets
            target_tensor = self.targets[self.mode][file_name]
            
            return image_tensor, target_tensor

        except KeyboardInterrupt:
            raise
        except Exception as e:
            if self.debug is False and self.verbose is True:
                print(repr(e))
            elif self.debug is True:
                raise
            return self.__getitem__(idx = np.random.randint(0, self.__len__()), layer = layer+1)

    def set_mode(self, mode: str):
        if not isinstance(mode, str):
            raise TypeError(f"mode must be of type 'str', but got type '{type(mode)}'.")
        if mode not in ["train", "val", "test"]:
            raise ValueError(f"mode should be one of ['train', 'val', 'test'], but was '{mode}'.")
        self.mode = mode

    def set_cached(self, target: str, value: bool = True):
        if target not in list(self.cached.keys()):
            raise ValueError(f"target should be one of {list(self.cached.keys())} but was '{target}'.")
        self.cached[target] = value

class LiTS_Segmentation_Dataset(torch.utils.data.Dataset):

    """
    Dataset for the LiTS challenge - this one loads data with targets intended for segmentation.
    data_dir is the parent_folder of the dataset.
    transforms should be an albumentations.Compose object or None.
    verbose controls the verbosity of the dataset.
    cache_data caches the data in the worker holding the dataset if set to True. Otherwise, read from disk.
    cache_mgr, if provided, should be a Manager, whose namespace contains a .data dict, which in turn has the
    "train", "val", and "test" keys. In this case, all workers cache to the memory of the Manager's process.
    """

    def __init__(
        self, 
        data_dir: str, 
        transforms: Callable = None, 
        verbose: bool = False, 
        cache_data: bool = False, 
        cache_mgr: object = None,
        debug: bool = False):

        super(LiTS_Segmentation_Dataset, self).__init__()
        
        self.data_dir = data_dir
        if not all([any([y for x in os.listdir(self.data_dir)]) for y in ["train", "val", "test"]]):
            raise ValueError(f"{self.data_dir} does not contain a 'train', 'val', and 'test' directory and is therefore probably wrong.")
        
        self.transforms = transforms
        if not callable(self.transforms) and not (self.transforms is None):
            raise TypeError(f"If transforms are provided, transforms must be a callable.")
        
        self.verbose = verbose
        
        self.cache_data = cache_data
        
        self.cache_mgr = cache_mgr
        if not (self.cache_mgr is None) and not all([x in list(self.cache_mgr.data.keys()) for x in ["train", "val", "test"]]):
            raise KeyError("Cache managers must be a Proxy for a dict object with the keys 'train', 'val', and 'test'.")
        if not (self.cache_mgr is None) and not all([x in list(self.cache_mgr.cached.keys()) for x in ["train", "val", "test"]]):
            raise KeyError("Cache managers must be a Proxy for a dict object with the keys 'train', 'val', and 'test'.")

        if self.cache_data is True and cache_mgr is not None:
            self.data = self.cache_mgr.data
            self.cached = self.cache_mgr.cached
        else:
            if self.cache_data is True:
                print("Warning: Caching without a cache manager leads to the cache being reset every epoch, unless num_workers is 0!")
            self.data = {
                "train": {},
                "val": {},
                "test": {}
                }
            self.cached = {
            "train": False,
            "val": False,
            "test": False
        }
        
        self.debug = debug

        self.mode = "train"

        self._make_file_names()
        if self.verbose:
            print("Dataset initialization complete.")

    def _make_file_names(self):
        
        train_d = os.path.join(self.data_dir, "train", "volumes")
        val_d = os.path.join(self.data_dir, "val", "volumes")
        test_d = os.path.join(self.data_dir, "test", "volumes")
        train_f = shallow_getFileList(path = train_d, fileType = ".png", pbar = self.verbose)
        val_f = shallow_getFileList(path = val_d, fileType = ".png", pbar = self.verbose)
        test_f = shallow_getFileList(path = test_d, fileType = ".png", pbar = self.verbose)
        self.file_names = {
            "train": {i: file for i, file in enumerate(train_f)},
            "val": {i: file for i, file in enumerate(val_f)},
            "test": {i: file for i, file in enumerate(test_f)}
        }

    def __len__(self, subset: str = None):

        if subset is None:
            return len(self.file_names[self.mode])
        else:
            return len(self.file_names[subset])

    def __getitem__(self, idx: int, layer: int = 0):

        try:
            # Find filename
            file_name = self.file_names[self.mode][idx]
            # If needed, load from disk
            if self.cache_data is False or self.cached[self.mode] is False:
                image_tensor, liver_tensor, lesion_tensor = lits_segmentation_loader(file = file_name)
            # If possible, try to load from cache
            elif self.cache_data is True and self.cached[self.mode] is True:
                try:
                    image_tensor = self.data[self.mode][file_name].to(dtype = torch.float32)
                    liver_tensor = self.data[self.mode][file_name+"_liver_mask"].to(dtype = torch.LongTensor)
                    lesion_tensor = self.data[self.mode][file_name+"_lesion_mask"].to(dtype = torch.LongTensor)
                except KeyError:
                    raise CacheError(f"Expected file {file_name} at index {idx} to be in cache, but got KeyError. File may be unavailable.")
            # Normalize to [0,1]
            image_tensor -= torch.min(image_tensor)
            image_tensor /= torch.max(image_tensor)
            # Save to cache, if required, cast to fp16 to save room
            if self.cache_data is True and self.cached[self.mode] is False:
                self.data[self.mode][file_name] = image_tensor.to(torch.float16)
                self.data[self.mode][file_name+"_liver_mask"] = image_tensor.to(torch.uint8)
                self.data[self.mode][file_name+"_lesion_mask"] = image_tensor.to(torch.uint8)

            # Transform training data
            targets = [liver_tensor, lesion_tensor]
            if self.transforms is not None and self.mode == "train":
                # Albumentations transforms require opencv arrays as input
                image_array = convert_tensor_to_opencv_array(image_tensor)
                targets = [convert_tensor_to_opencv_array(t) for t in targets]

                tf = self.transforms(image = image_array, masks = targets)
                image_array, targets = tf["image"], tf["masks"]

                image_tensor = convert_opencv_array_to_tensor(image_array)
                targets = [convert_opencv_array_to_tensor(t) for t in targets]
            
            return image_tensor, targets

        except KeyboardInterrupt:
            raise
        except Exception as e:
            if self.debug is False and self.verbose is True:
                print(repr(e))
            elif self.debug is True:
                raise
            return self.__getitem__(idx = np.random.randint(0, self.__len__()), layer = layer+1)

    def set_mode(self, mode: str):
        if not isinstance(mode, str):
            raise TypeError(f"mode must be of type 'str', but got type '{type(mode)}'.")
        if mode not in ["train", "val", "test"]:
            raise ValueError(f"mode should be one of ['train', 'val', 'test'], but was '{mode}'.")
        self.mode = mode

    def set_cached(self, target: str, value: bool = True):
        if target not in list(self.cached.keys()):
            raise ValueError(f"target should be one of {list(self.cached.keys())} but was '{target}'.")
        self.cached[target] = value

