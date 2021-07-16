import tensorflow as tf
import numpy as np
import cv2
import os

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]

CHARS_DICT = {char: i for i, char in enumerate(CHARS)}

# 图片转换操作，注意：cv2读取的图片为GBR格式,转为RGB格式
def transform(img):
    img = img.astype(np.float32)
    b, g, r= cv2.split(img)
    img = cv2.merge([r, g, b])
    return img

# 检查异常label
def check(label):
    if label[2] != CHARS_DICT['D'] and label[2] != CHARS_DICT['F'] \
            and label[-1] != CHARS_DICT['D'] and label[-1] != CHARS_DICT['F']:
        print("Error label, please check it")
        return False
    else:
        return True

# 生成tf.data.Dataset数据集用于训练
def load_data(img_dir, lpr_len, batch_size, img_size=(94, 24)):
    imgs_num = len(os.listdir(img_dir))
    
    imgs = []
    labels = []
    for i, img in enumerate(os.listdir(img_dir)): 
        # 生成label数据
        label = []
        lpr_name, suffix = os.path.splitext(img)
        for c in lpr_name:
            label.append(CHARS_DICT[c])
        if len(label) == 8:
            if check(label) == False:
                print(img)
                assert 0, "Error label!"
        # 生成image数据
        img = cv2.imread(os.path.join(img_dir, img))
        img = transform(img)
        height, width, _ = img.shape
        if height != img_size[1] or width != img_size[0]:
                img = cv2.resize(img, img_size)
        
        imgs.append(img)
        labels.append(label)
    
    dataset = tf.data.Dataset.from_tensor_slices((imgs, labels))
    return dataset, imgs_num

# 由dense生成sparse
def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    # 自动寻找序列的最大长度，形状为：batch_size * max_len
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return tf.SparseTensor(indices=indices, values=values, dense_shape=shape)

