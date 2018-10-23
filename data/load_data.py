from common import *

from torch.utils.data.dataset import Dataset

#----------------------------------------
def null_augment(image,label,index):
    cache = Struct(image = image.copy(), mask = mask.copy())
    return image,label,index, cache


def null_collate(batch):

    batch_size = len(batch)
    cache = []
    input = []
    truth = []
    index = []
    for b in range(batch_size):
        input.append(batch[b][0])
        truth.append(batch[b][1])
        index.append(batch[b][2])
        cache.append(batch[b][3])
    input = torch.from_numpy(np.array(input)).float().unsqueeze(1)

    if truth[0]!=[]:
        truth = torch.from_numpy(np.array(truth)).float().unsqueeze(1)

    return input, truth, index, cache


#----------------------------------------
class TsgDataset(Dataset):

    def __init__(self, split, augment=null_augment, mode='train', scale='pad'):
        super(TsgDataset, self).__init__()
        self.split   = split
        self.mode   = mode
        self.augment = augment
        self.scale = scale

        # split_file =  DATA_DIR + '/split/' + split
        split_file =  PROJECT_DIR + '/data/split/' + split
        lines = read_list_from_file(split_file)

        self.ids    = []
        self.images = []
        for l in lines:
            folder, name = l.split('/')
            image_file = DATA_DIR + '/' + folder + '/images/' + name +'.png'
            image = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
            # image = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE).astype(np.float16)/255
            self.images.append(image)
            self.ids.append(name)

        self.masks  = []
        if self.mode in ['train','valid']:
            for l in lines:
                folder, file = l.split('/')
                mask_file  = DATA_DIR + '/' + folder + '/masks/' + file +'.png'
                mask  = cv2.imread(mask_file,cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
                # mask  = cv2.imread(mask_file,cv2.IMREAD_GRAYSCALE).astype(np.float16)/255
                self.masks.append(mask)
        elif self.mode in ['test']:
            self.masks  = [[] for l in lines]

        #-------
        df = pd.read_csv(DATA_DIR + '/depths.csv')
        df = df.set_index('id')
        self.zs = df.loc[self.ids].z.values

        #-------
        print('\tTsgDataset')
        print('\tsplit            = %s'%split)
        print('\tlen(self.images) = %d'%len(self.images))
        # print('\timage shape = %s' % str(self.images[0].shape))
        print('')


    def __getitem__(self, index):
        image = self.images[index]
        mask  = self.masks[index]

        # print(self.augment)
        try:
            return self.augment(image, mask, index, scale=self.scale)
        except:
            return self.augment(image, mask, index)

    def __len__(self):
        return len(self.images)
