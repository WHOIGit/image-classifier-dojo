import os
import argparse
import random
import sys

from torch.utils.data import random_split

def argparse_init():
    parser = argparse.ArgumentParser(description='Create DATASET_labels.list, DATASET_training.list, DATASET_validation.list from local directories')
    parser.add_argument('--name', metavar='DATASET', required=True)
    parser.add_argument('--target', metavar='DIR', required=True, help='Directory with class-label subfolders and images')
    #data.add_argument('--class-config', metavar=('CSV', 'COL'), nargs=2, help='Skip and combine classes as defined by column COL of a special CSV configuration file')
    parser.add_argument('--split', metavar='RATIO', type=float, default=0.8, help='Ratio of training-to-validation datasplit, as a float. Eg. "0.8" means 80% of the TARGET data will be put into TRAINFILE, leaving 20% for VALFILE')
    #parser.add_argument('--balanced', action='store_true')
    parser.add_argument('--seed', type=int, help='Set a specific seed for deterministic reproducability')
    parser.add_argument('--class-min', metavar='MIN', type=int, default=2, help='Exclude classes with fewer than MIN instances. Default is 2')
    parser.add_argument('--class-max', metavar='MAX', type=int, help='Limit classes to a MAX number of instances')

    out = parser.add_argument_group(title='Output Options')
    out.add_argument('--outdir', default='.')
    out.add_argument('--labelfile', default='{DATASET}_labels.list')
    out.add_argument('--trainfile', default='{DATASET}_training.list')
    out.add_argument('--valfile', default='{DATASET}_validation.list')
    return parser


def fetch_images_perclass(src:str):
    """ folders in src are the classes """
    import torchvision
    if os.path.isdir(src):
        classes = [d.name for d in os.scandir(src) if d.is_dir()]
        classes.sort()

        images_perclass = {}
        for subdir in classes:
            files = os.listdir(os.path.join(src, subdir))
            files = sorted([f for f in files if torchvision.datasets.folder.is_image_file(f)])
            images_perclass[subdir] = [os.path.join(src, subdir, i) for i in files]
        return images_perclass


def limit_images_perclass(images_perclass, minimum_images_per_class=2, maximum_images_per_class=None):

    # CLASS MINIMUM CUTTOFF
    images_perclass__minthresh = {label: images for label, images in images_perclass.items() if
                                  len(images) >= minimum_images_per_class}
    classes_ignored = sorted(set(images_perclass.keys()) - set(images_perclass__minthresh.keys()))
    classes_removed_for_too_few_samples = [(c, len(images_perclass[c])) for c in classes_ignored]
    classes = sorted(images_perclass__minthresh.keys())

    # CLASS MAXIMUM LIMITING
    if maximum_images_per_class:
        assert maximum_images_per_class >= minimum_images_per_class
        images_perclass__maxlimited = {
            label: sorted(random.sample(images, maximum_images_per_class)) \
                if maximum_images_per_class < len(images) \
                else images \
            for label, images in images_perclass__minthresh.items()
        }
        images_perclass__final = images_perclass__maxlimited
        classes_limited_for_too_many_samples = \
            [c for c in classes if len(images_perclass__maxlimited[c]) < len(images_perclass__minthresh[c])]
    else:
        images_perclass__final = images_perclass__minthresh
        classes_limited_for_too_many_samples = None

    # sort perclass images internally, just because its nice.
    images_perclass__final = {label: sorted(images) for label, images in images_perclass__final.items()}
    return images_perclass__final, classes_removed_for_too_few_samples, classes_limited_for_too_many_samples


def balanced_split(images_perclass, ratio):
    classes = list(images_perclass.keys())
    total_image_count = sum(len(vals) for vals in images_perclass.values())
    d1_perclass = {}
    d2_perclass = {}
    for class_label, images in images_perclass.items():
        # 1) determine output lengths
        d1_len = int(ratio * len(images) / 100 + 0.5)
        if d1_len == len(images):
            # make sure that at least one image gets put in d2
            d1_len -= 1

        # 2) split images as per distribution
        d1_images = random.sample(images, d1_len)
        d2_images = sorted(list(set(images) - set(d1_images)))
        assert len(d1_images) + len(d2_images) == len(images)

        # 3) put images into perclass_sets at the right class
        d1_perclass[class_label] = d1_images
        d2_perclass[class_label] = d2_images

    # 4) create and return new datasets
    assert set(d1_perclass.keys()) == set(d2_perclass.keys()), \
        f'd1-d2_classes:{set(d1_perclass.keys()) - set(d2_perclass.keys())}, d2-d1_classes:{set(d2_perclass.keys()) - set(d1_perclass.keys())}'  # possibly fails due to edge case thresholding?
    assert set(classes) == set(
        d1_perclass.keys()), f'd0-d1_classes:{set(classes) - set(d1_perclass.keys())}'
    training_samples = [(src, classes.index(label)) for label, sources in d1_perclass.items() for src in
                        sources]
    validation_samples = [(src, classes.index(label)) for label, sources in d2_perclass.items() for src in
                          sources]

    assert len(training_samples) + len(validation_samples) == total_image_count  # make sure we don't lose any images somewhere
    return training_samples, validation_samples


def main(args:argparse.Namespace):
    if not args.seed:
        args.seed = random.randrange(sys.maxsize)
    print(f'Seed is {args.seed}')
    random.seed(args.seed)

    images_perclass = fetch_images_perclass(args.target)
    images_perclass, classes_ignored_for_too_few_samples, classes_limited_for_too_many_samples = limit_images_perclass(images_perclass)
    classes = list(images_perclass.keys())
    training_samples, validation_samples = balanced_split(images_perclass, args.split)

    if classes_ignored_for_too_few_samples:
        msg = '\n{} out of initial {} classes ignored from --class-min {}, PRE-SPLIT'
        msg = msg.format(len(classes_ignored_for_too_few_samples), len(classes+classes_ignored_for_too_few_samples), args.class_min)
        classes_ignored = [f'({l:2}) {c}' for c,l in classes_ignored_for_too_few_samples]
        print('\n    '.join([msg]+classes_ignored))
    if classes_limited_for_too_many_samples:
        msg = '\n{} out of {} classes ignored from --class-max {}, PRE-SPLIT'
        msg = msg.format(len(classes_limited_for_too_many_samples), len(classes), args.class_max)
        #classes_limited= [f'({l:6}) {c}' for c,l in classes_limited_for_too_many_samples]
        print('\n    '.join([msg] + classes_limited_for_too_many_samples))

    # Output Files
    labelfile, trainfile, valfile = [os.path.join(args.outdir,fname) for fname in (args.labelfile,args.trainfile,args.valfile)]
    labelfile, trainfile, valfile = [fname.format(DATASET=args.name) for fname in (labelfile,trainfile,valfile)]
    [os.makedirs(os.path.dirname(path), exist_ok=True) for path in (labelfile,trainfile,valfile)]
    with open(labelfile,'w') as labf, open(trainfile,'w') as trnf, open(valfile,'w') as valf:
        labf.write('\n'.join(classes))
        trnf.write('\n'.join([f'{idx:>03} {src}' for src,idx in training_samples]))
        valf.write('\n'.join([f'{idx:>03} {src}' for src, idx in training_samples]))


if __name__ == '__main__':
    parser = argparse_init()
    args = parser.parse_args()
    main(args)

