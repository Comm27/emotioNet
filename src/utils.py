import os
import pandas as pd
import numpy as np

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def text_to_numpy_arr(txt):
    txt_arr = txt.split()
    return np.array(txt_arr).astype('uint8')


def get_data_from_df(df, shape=(48, 48, 1)):
    train = df.Usage == 'Training'
    test = df.Usage != 'Training'
    
    train_labels = df[train].emotion.to_numpy()
    test_labels = df[test].emotion.to_numpy()    
    
    tr = df[train].pixels
    tst = df[test].pixels
    
    train_imgs = [text_to_numpy_arr(x).reshape(shape) for x in tr]
    test_imgs = [text_to_numpy_arr(x).reshape(shape) for x in tst]
    
    train_imgs = np.array(train_imgs)
    test_imgs = np.array(test_imgs)
    return (train_imgs, train_labels), (test_imgs, test_labels)


def arr_to_img(img_arr, img_labels, save_folder, ext='.jpg'):
    from PIL import Image
    for i, (img, label) in enumerate(zip(img_arr, img_labels)):
        class_name = class_labels[label]
        img_folder = os.path.join(save_folder, class_name)
        img_name = str(i).zfill(6) + ext

        img_fullPath = os.path.join(img_folder, img_name)

        img = np.squeeze(img, axis=2)
        image = Image.fromarray(img)
        image.save(img_fullPath)


def csv_2_imgs(csv_filepath, main_folder_name='data', ext='.jpg'):
    df = pd.read_csv(csv_filepath)
    
    (train_imgs, train_labels), (test_imgs, test_labels) = get_data_from_df(df)

    main_folder_name = os.path.join('..', main_folder_name)
    train_folder = os.path.join(main_folder_name, 'train')
    test_folder = os.path.join(main_folder_name, 'test')

    # create folders if they don't exist
    if not os.path.exists(main_folder_name): os.mkdir(main_folder_name)
    if not os.path.exists(train_folder): os.mkdir(train_folder)
    if not os.path.exists(test_folder): os.mkdir(test_folder)
    
    for class_name in class_labels:
        path_tr = os.path.join(train_folder, class_name)
        path_tst = os.path.join(test_folder, class_name)
        if not os.path.exists(path_tr): os.mkdir(path_tr)
        if not os.path.exists(path_tst): os.mkdir(path_tst)
    
    # save images to folders
    arr_to_img(train_imgs, train_labels, train_folder)
    arr_to_img(test_imgs, test_labels, test_folder)


if __name__ == '__main__':
    csv_path = os.path.join('..', 'dataset', 'fer2013', 'fer2013.csv')
    csv_2_imgs(csv_path)
