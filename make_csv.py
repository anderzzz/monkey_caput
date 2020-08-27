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
    parser.add_argument('--out', dest='out', default='out.csv',
                        help='Path to output file')

    args = parser.parse_args()

    return args.img_root, args.out

def dir_content(path):
    return [x for x in os.listdir(path) if x[0] != '.']

def main(args):

    file_root, out_path = parse_cmd(args)
    df_start = pd.read_csv('{}/toc.csv'.format(file_root))
    df_start = df_start.iloc[:, 1:]
    all_dfs = []
    for id, row in df_start.iterrows():
        family_label = row.Family
        genus_content = dir_content('{}/{}'.format(file_root, family_label))
        for genus in genus_content:
            species_content = dir_content('{}/{}/{}'.format(file_root, family_label, genus))
            for species in species_content:
                img_files = dir_content('{}/{}/{}/{}'.format(file_root, family_label, genus, species))
                row_items = [(genus, species, x) for x in img_files]

                # DataFrame with the higher order indeces for the biological species
                row_higher = pd.DataFrame(row).T.iloc[:, :-2]

                # DataFrame with the lower order indeces for the biological species plus the image name and index
                img_details = pd.DataFrame(row_items, columns=['Genus', 'Species', 'ImageName'],
                                           index=pd.RangeIndex(start=0, stop=len(row_items), name='InstanceIndex'))

                # Merge the higher and lower order indexed data, keep only the ImageName as column
                fungi_higher = row_higher.reindex(img_details.index, method='nearest')
                df_1 = pd.concat([img_details, fungi_higher], axis=1)
                new_index = df_1.columns.drop(['ImageName'])
                df_expanded = df_1.set_index(list(new_index.to_numpy()), append=True)
                df_expanded = df_expanded.reorder_levels([3, 4 ,5 ,6, 7, 8, 1, 2, 0])

                all_dfs.append(df_expanded)

    df_all = pd.concat(all_dfs)
    df_all.to_csv(out_path)


if __name__ == '__main__':
    main(sys.argv[1:])