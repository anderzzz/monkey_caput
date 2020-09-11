'''Main script to setup and run training

'''
import sys
import time
from numpy.random import shuffle

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

from fungiimg import FungiImg, StandardTransform, RawData
from model_init import initialize_model
from trainer import train_model

def main(run_label, f_out,
         raw_csv_toc, raw_csv_root, transform_key, label_key, f_test,
         loader_batch_size, num_workers, n_epochs,
         model_label, use_pretrained,
         save_file_name):

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
    if label_key == 'Kantarell vs Fluesvamp':
        label_keys = ('Family == "Cantharellaceae"', 'Family == "Amanitaceae"')
    elif label_key == 'Champignon vs Fluesvamp':
        label_keys = ('Genus == "Agaricus"', 'Genus == "Amanita"')
    elif label_key is None:
        label_keys = None
    else:
        raise ValueError('Unknown label_key: {}'.format(label_key))

    if transform_key == 'standard_300':
        transform = StandardTransform(300, to_tensor=True, normalize=False)
    else:
        raise ValueError('Unknown transform_key: {}'.format(transform_key))

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
    dataset_sizes = {'train' : len(dataset_train), 'test' : len(dataset_test)}
    print (dataset_sizes)
    print (dataset_train.label_semantics)
    print (dataset_train.n_instance_genus)
    print (dataset_train.n_instance_family)

    #
    # Define the model
    #
    if label_key == 'Kantarell vs Fluesvamp':
        num_classes = dataset_train.n_family
    if label_key == 'Champignon vs Fluesvamp':
        num_classes = dataset_train.n_genus
    elif label_key is None:
        num_classes = dataset_train.n_species
    else:
        raise ValueError('Unknown label_key: {}'.format(label_key))

    model, input_size = initialize_model(model_label, num_classes, use_pretrained)

    #
    # Define criterion and optimizer and scheduler
    #
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    #
    # Train
    #
    is_inception = 'inception' in model_label
    best_model = train_model(model, criterion, optimizer, exp_lr_scheduler,
                             n_epochs, dataloaders, dataset_sizes,
                             is_inception)

    with open(save_file_name, 'w') as f_out:
        torch.save(best_model, f_out)

if __name__ == '__main__':

#    main('Test Run', sys.stdout,
#         '../../Desktop/Fungi/toc_full.csv', '../../Desktop/Fungi',
#         'standard_300', 'Kantarell vs Fluesvamp', 0.10,
#         8, 1, 21,
#         'inception_v3', True,
#         'save_me.pkl')

    main('Binary Tougher', sys.stdout,
         '../../Desktop/Fungi/toc_full.csv', '../../Desktop/Fungi',
         'standard_300', 'Champignon vs Fluesvamp', 0.10,
         8, 1, 21,
         'inception_v3', True,
         'save_me_champ_flue.pkl')
