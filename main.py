'''Main script to setup and run training

'''
import sys
import time
from numpy.random import shuffle
from torch.utils.data import DataLoader

from fungiimg import FungiImg, StandardTransform, RawData
from model_init import FungiModel

def main(run_label, f_out,
         raw_csv_toc, raw_csv_root, transform_key, label_key, f_test,
         loader_batch_size, num_workers,
         model_label):

    # Print all run inputs to file
    inp_args = locals()
    the_time = time.localtime()
    print('Run at {}/{}/{} {}:{}:{} with arguments:'.format(the_time.tm_year, the_time.tm_mon, the_time.tm_mday,
                                                            the_time.tm_hour, the_time.tm_min, the_time.tm_sec), file=f_out)
    for inp_key, inp_val in inp_args.items():
        print ('{}: {}'.format(inp_key, inp_val), file=f_out)

    #
    # Define the dataset and dataloader, train and test, using the string short-hand
    #
    if transform_key == 'standard_300':
        transform = StandardTransform(300, to_tensor=True, normalize=False)
    else:
        raise ValueError('Unknown transform_key: {}'.format(transform_key))

    if label_key == 'Kantarell vs Fluesvamp':
        label_keys =('Genus == "Cantharellus"', 'Genus == "Amanita"')
    elif label_key is None:
        label_keys = None
    else:
        raise ValueError('Unknown label_key: {}'.format(label_key))

    all_ids = list(range(RawData.N_ROWS.value))
    shuffle(all_ids)
    n_test = int(RawData.N_ROWS.value * f_test)
    test_ids = all_ids[:n_test]
    train_ids = all_ids[n_test:]

    dataset_train = FungiImg(csv_file=raw_csv_toc, root_dir=raw_csv_root,
                             iselector=train_ids, transform=transform,
                             label_keys=label_keys)
    dataset_test = FungiImg(csv_file=raw_csv_toc, root_dir=raw_csv_root,
                            iselector=test_ids, transform=transform,
                            label_keys=label_keys)
    dataloaders = {'train' : DataLoader(dataset_train, batch_size=loader_batch_size,
                                        shuffle=True, num_workers=num_workers),
                   'test' : DataLoader(dataset_test, batch_size=loader_batch_size,
                                       shuffle=False, num_workers=num_workers)}

    #
    # Define the model
    #
    model = FungiModel(label=model_label)

if __name__ == '__main__':

    main('Test Run', sys.stdout,
         '../../Desktop/Fungi/toc_full.csv', '../../Desktop/Fungi',
         'standard_300', 'Kantarell vs Fluesvamp', 0.05,
         4, 1,
         'inception_v3')