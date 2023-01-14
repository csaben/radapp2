#POPULATE IMG_DATASET DIRECTORY
from dataset import *
from torchvision.transforms import ToPILImage
from config import *
# from glioma_utils import *
from PIL import Image

def main():
    # dir_c = ds_helper(CONTROL)
    # dir_t = ds_helper(TUMOR)
    # make_img_dir(dir_t, "TUMOR")
    # make_img_dir(dir_c, "CONTROL")
    pass



def avg_img(dcm_dir):
    #grab each dcm, convert to pixel, average together
    pixel_data_list = []
    # Iterate over all .dcm files in the directory
    for file in os.listdir(dcm_dir):
        # Load .dcm file
        dcm_file = pydicom.dcmread(os.path.join(dcm_dir, file))
        # Add pixel data to list
        pixel_data_list.append(dcm_file.pixel_array)

    # Calculate average of pixel data
    average_pixel_data = np.mean(pixel_data_list, axis=0)
    return average_pixel_data

def find_dcm_dirs(path):
    dcm_dirs = []
    for root, dirs, files in os.walk(path):
        #check if root contains a capital "S#"
        pattern = r'S[0-9]/[a-zA-Z]'
        regexp = re.compile(pattern)
        if regexp.search(root):
            dcm_dirs.append(root)
            # print(glob.glob(f'{root}/*.dcm'))
    return dcm_dirs


def ds_helper(path):
    dcm_dirs = find_dcm_dirs(path)
    return(dcm_dirs)


def make_img_dir(dcm_dir, pathology):
    for i,directory in enumerate(dcm_dir):
        image=avg_img(directory)
        image = torch.from_numpy(image)
        image = ToPILImage()(image)
        image.save('../input/img_dataset/'+pathology+f'/avg_img_{i}.jpg', format='JPEG')



if __name__ == '__main__':
    main()

