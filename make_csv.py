'''Helper script to make CSV with data annotations

'''
import sys
import os
import pandas as pd
import argparse

def parse_cmd(cmd_line):

    parser = argparse.ArgumentParser(description='Generate Image Meta Data')
    parser.add_argument('--img-root', dest='img_root',
                        help='Path to image folders root')

    args = parser.parse_args()

    return args.img_root

def dir_content(path):
    return [x for x in os.listdir(path) if x[0] != '.']

def main(args):

    file_root = parse_cmd(args)
    df_start = pd.read_csv('{}/toc.csv'.format(file_root))
    df_start = df_start.iloc[:, 1:]
    for id, row in df_start.iterrows():
        family_label = row.Family
        genus_content = dir_content('{}/{}'.format(file_root, family_label))
        for genus in genus_content:
            species_content = dir_content('{}/{}/{}'.format(file_root, family_label, genus))
            for species in species_content:
                img_files = dir_content('{}/{}/{}/{}'.format(file_root, family_label, genus, species))
                row_items = [(genus, species, x) for x in img_files]

                print (pd.DataFrame(row_items, columns=['Genus', 'Species', 'ImageName']))
                raise RuntimeError


if __name__ == '__main__':
    main(sys.argv[1:])