import numpy as np
from PIL import Image
import torch


def try_gpu():
    use_cuda = torch.cuda.is_available()
    return torch.device("cuda:0" if use_cuda else "cpu")


def nms(boxes, overlap_threshold=0.5, mode="union"):
    """非极大值抑制（Non-Maximum Suppression），用于去除重叠度高的边界框

    - boxes : numpy.ndarray ，形状 [n, 5]
        - 含义: 边界框数组，每行格式为 (xmin, ymin, xmax, ymax, score)
        - 数据类型: float32
        - n: 边界框数量
    - overlap_threshold : float ，默认值 0.5
        - 含义: 重叠度阈值，超过此值的边界框将被抑制
        - 数据类型: float
        - 取值范围: 0.0-1.0
    - mode : str ，可选值 'union' 或 'min' ，默认值 'union'
        - 含义: 重叠度计算模式
        - 'union' : 使用IoU（Intersection over Union）
        - 'min' : 使用交集与较小框面积的比值
    """

    # if there are no boxes, return the empty list
    if len(boxes) == 0:
        return []

    # list of picked indices
    pick = []

    # grab the coordinates of the bounding boxes
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]

    area = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    ids = np.argsort(score)  # in increasing order

    while len(ids) > 0:

        # grab index of the largest value
        last = len(ids) - 1
        i = ids[last]
        pick.append(i)

        # compute intersections
        # of the box with the largest score
        # with the rest of boxes

        # left top corner of intersection boxes
        ix1 = np.maximum(x1[i], x1[ids[:last]])
        iy1 = np.maximum(y1[i], y1[ids[:last]])

        # right bottom corner of intersection boxes
        ix2 = np.minimum(x2[i], x2[ids[:last]])
        iy2 = np.minimum(y2[i], y2[ids[:last]])

        # width and height of intersection boxes
        w = np.maximum(0.0, ix2 - ix1 + 1.0)
        h = np.maximum(0.0, iy2 - iy1 + 1.0)

        # intersections' areas
        inter = w * h
        if mode == "min":
            overlap = inter / np.minimum(area[i], area[ids[:last]])
        elif mode == "union":
            # intersection over union (IoU)
            overlap = inter / (area[i] + area[ids[:last]] - inter)

        # delete all boxes where overlap is too big
        ids = np.delete(
            ids, np.concatenate([[last], np.where(overlap > overlap_threshold)[0]])
        )

    return pick


def convert_to_square(bboxes):
    """
    将任意形状的边界框转换为正方形

    Arguments:
        bboxes : numpy.ndarray ，形状 [n, 5]
        - 含义: 边界框数组，每行格式为 (xmin, ymin, xmax, ymax, score)
        - 数据类型: float32
        - n: 边界框数量

    Returns:
        - 类型 : numpy.ndarray ，形状 [n, 5]
        - 含义 : 转换为正方形的边界框数组
        - 结构 : 与输入相同的格式，但所有边界框都变为正方形
    
    #关键实现逻辑
    1. 提取坐标 : 从边界框中提取xmin, ymin, xmax, ymax
    2. 计算尺寸 : 计算每个边界框的宽度和高度
    3. 确定最大边 : 取宽度和高度的最大值作为正方形边长
    4. 中心对齐 : 以原边界框中心为基准，向四周扩展到正方形尺寸
    5. 计算新坐标 :
    - 新xmin = 原xmin + 原宽度/2 - 最大边/2
    - 新ymin = 原ymin + 原高度/2 - 最大边/2
    - 新xmax = 新xmin + 最大边 - 1
    - 新ymax = 新ymin + 最大边 - 1

    """

    square_bboxes = np.zeros_like(bboxes)
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    h = y2 - y1 + 1.0
    w = x2 - x1 + 1.0
    max_side = np.maximum(h, w)
    square_bboxes[:, 0] = x1 + w * 0.5 - max_side * 0.5
    square_bboxes[:, 1] = y1 + h * 0.5 - max_side * 0.5
    square_bboxes[:, 2] = square_bboxes[:, 0] + max_side - 1.0
    square_bboxes[:, 3] = square_bboxes[:, 1] + max_side - 1.0
    return square_bboxes


def calibrate_box(bboxes, offsets):
    """Transform bounding boxes to be more like true bounding boxes.
    'offsets' is one of the outputs of the nets.

    - bboxes : numpy.ndarray ，形状 [n, 5]
        - 含义: 原始边界框数组，每行格式为 (xmin, ymin, xmax, ymax, score)
        - 数据类型: float32
        - n: 边界框数量
    
    - offsets : numpy.ndarray ，形状 [n, 4]
        - 含义: 边界框偏移量数组，每行格式为 (tx1, ty1, tx2, ty2)
        - 数据类型: float32
        - 偏移量含义:
        - tx1 : xmin的偏移量（相对于宽度的比例）
        - ty1 : ymin的偏移量（相对于高度的比例）
        - tx2 : xmax的偏移量（相对于宽度的比例）
        - ty2 : ymax的偏移量（相对于高度的比例）

    ### 关键实现逻辑
        1. 提取坐标和尺寸 : 从边界框中提取xmin, ymin, xmax, ymax，计算宽度和高度
        2. 维度扩展 : 将宽度和高度扩展为列向量，便于广播运算
        3. 计算平移量 :
        - translation = [w, h, w, h] * offsets
        - 即: translation = [w*tx1, h*ty1, w*tx2, h*ty2]
        4. 应用偏移 : 将平移量加到原始边界框坐标上
        5. 公式解释 :
        - x1_new = x1 + w * tx1
        - y1_new = y1 + h * ty1
        - x2_new = x2 + w * tx2
        - y2_new = y2 + h * ty2
    """
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0
    w = np.expand_dims(w, 1)
    h = np.expand_dims(h, 1)

    # this is what happening here:
    # tx1, ty1, tx2, ty2 = [offsets[:, i] for i in range(4)]
    # x1_true = x1 + tx1*w
    # y1_true = y1 + ty1*h
    # x2_true = x2 + tx2*w
    # y2_true = y2 + ty2*h
    # below is just more compact form of this

    # are offsets always such that
    # x1 < x2 and y1 < y2 ?

    translation = np.hstack([w, h, w, h]) * offsets
    bboxes[:, 0:4] = bboxes[:, 0:4] + translation
    return bboxes


def get_image_boxes(bounding_boxes, img, size=24):
    """从原始图像中裁剪出边界框对应的图像区域，并调整为指定尺寸

    ### 输入参数
    - bounding_boxes : numpy.ndarray ，形状 [n, 5]
    - 含义: 边界框数组，每行格式为 (xmin, ymin, xmax, ymax, score)
    - 数据类型: float32
    - n: 边界框数量

    - img : PIL.Image.Image
    - 含义: 原始输入图像
    - 数据类型: PIL.Image对象

    - size : int ，默认值 24
    - 含义: 输出图像的尺寸（宽度和高度相同）
    - 数据类型: int
    - 常用值: 24（R-Net输入）或48（O-Net输入）

    ### 返回值
    - 类型 : numpy.ndarray ，形状 [n, 3, size, size]
    - 含义 : 裁剪并预处理后的图像数组
    - 结构 :
    - 第一维: 边界框数量
    - 第二维: 通道数（RGB）
    - 第三、四维: 图像尺寸（size × size）
    - 数据类型 : float32

    ### 关键实现逻辑
    1. 边界框校正 : 调用 correct_bboxes() 处理超出图像边界的边界框
    2. 初始化输出数组 : 创建零数组存储处理后的图像
    3. 逐个处理边界框 :
    - 从原始图像中裁剪对应区域
    - 处理边界框部分超出图像边界的情况
    - 将裁剪的图像调整为指定尺寸（双线性插值）
    - 对图像进行预处理
    4. 预处理 : 调用 preprocess() 函数进行标准化处理
    5. 返回结果 : 返回所有处理后的图像数组
    """

    num_boxes = len(bounding_boxes)
    width, height = img.size

    [dy, edy, dx, edx, y, ey, x, ex, w, h] = correct_bboxes(
        bounding_boxes, width, height
    )
    img_boxes = np.zeros((num_boxes, 3, size, size), "float32")

    for i in range(num_boxes):
        img_box = np.zeros((h[i], w[i], 3), "uint8")

        img_array = np.asarray(img, "uint8")
        img_box[dy[i] : (edy[i] + 1), dx[i] : (edx[i] + 1), :] = img_array[
            y[i] : (ey[i] + 1), x[i] : (ex[i] + 1), :
        ]

        # resize
        img_box = Image.fromarray(img_box)
        img_box = img_box.resize((size, size), Image.BILINEAR)
        img_box = np.asarray(img_box, "float32")

        img_boxes[i, :, :, :] = preprocess(img_box)

    return img_boxes


def correct_bboxes(bboxes, width, height):
    """校正超出图像边界的边界框，确保所有边界框都在图像范围内

    ### 输入参数
    - bboxes : numpy.ndarray ，形状 [n, 5]
    - 含义: 边界框数组，每行格式为 (xmin, ymin, xmax, ymax, score)
    - 数据类型: float32
    - n: 边界框数量

    - width : float
    - 含义: 图像宽度
    - 数据类型: float

    - height : float
    - 含义: 图像高度
    - 数据类型: float

    Returns:
        dy, dx, edy, edx: a int numpy arrays of shape [n],
            coordinates of the boxes with respect to the cutouts.
        y, x, ey, ex: a int numpy arrays of shape [n],
            corrected ymin, xmin, ymax, xmax.
        h, w: a int numpy arrays of shape [n],
            just heights and widths of boxes.

        in the following order:
            [dy, edy, dx, edx, y, ey, x, ex, w, h].
    """

    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w, h = x2 - x1 + 1.0, y2 - y1 + 1.0
    num_boxes = bboxes.shape[0]

    # 'e' stands for end
    # (x, y) -> (ex, ey)
    x, y, ex, ey = x1, y1, x2, y2

    # we need to cut out a box from the image.
    # (x, y, ex, ey) are corrected coordinates of the box
    # in the image.
    # (dx, dy, edx, edy) are coordinates of the box in the cutout
    # from the image.
    dx, dy = np.zeros((num_boxes,)), np.zeros((num_boxes,))
    edx, edy = w.copy() - 1.0, h.copy() - 1.0

    # if box's bottom right corner is too far right
    ind = np.where(ex > width - 1.0)[0]
    edx[ind] = w[ind] + width - 2.0 - ex[ind]
    ex[ind] = width - 1.0

    # if box's bottom right corner is too low
    ind = np.where(ey > height - 1.0)[0]
    edy[ind] = h[ind] + height - 2.0 - ey[ind]
    ey[ind] = height - 1.0

    # if box's top left corner is too far left
    ind = np.where(x < 0.0)[0]
    dx[ind] = 0.0 - x[ind]
    x[ind] = 0.0

    # if box's top left corner is too high
    ind = np.where(y < 0.0)[0]
    dy[ind] = 0.0 - y[ind]
    y[ind] = 0.0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, w, h]
    return_list = [i.astype("int32") for i in return_list]

    return return_list


def preprocess(img):
    """Preprocessing step before feeding the network.

    Arguments:
        img: a float numpy array of shape [h, w, c].

    Returns:
        a float numpy array of shape [1, c, h, w].
    """
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = (img - 127.5) * 0.0078125
    return img


def generate_bboxes(probs, offsets, scale, threshold):
    """Generate bounding boxes at places
    where there is probably a face.

    Arguments:
        probs: a float numpy array of shape [n, m].
        offsets: a float numpy array of shape [1, 4, n, m].
        scale: a float number,
            width and height of the image were scaled by this number.
        threshold: a float number.

    Returns:
        a float numpy array of shape [n_boxes, 9]
    """

    # applying P-Net is equivalent, in some sense, to
    # moving 12x12 window with stride 2
    stride = 2
    cell_size = 12

    # indices of boxes where there is probably a face
    # inds[0]: 满足条件的元素的行索引
    # inds[1]: 满足条件的元素的列索引
    # 输出: (array([0, 0, 1, 1, 2, 2, 3, 3]), array([1, 3, 0, 2, 2, 3, 0, 3]))
    inds = np.where(probs > threshold)

    if inds[0].size == 0:
        return np.array([])

    # transformations of bounding boxes
    tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]
    # they are defined as:
    # w = x2 - x1 + 1
    # h = y2 - y1 + 1
    # x1_true = x1 + tx1*w
    # x2_true = x2 + tx2*w
    # y1_true = y1 + ty1*h
    # y2_true = y2 + ty2*h

    offsets = np.array([tx1, ty1, tx2, ty2]) # 形状 ： [4, n_boxes]
    score = probs[inds[0], inds[1]] # 形状 ： [n_boxes]

    # P-Net is applied to scaled images
    # so we need to rescale bounding boxes back
    bounding_boxes = np.vstack(
        [
            np.round((stride * inds[1] + 1.0) / scale), # 形状 ： [n_boxes]
            np.round((stride * inds[0] + 1.0) / scale), # 形状 ： [n_boxes]
            np.round((stride * inds[1] + 1.0 + cell_size) / scale), # 形状 ： [n_boxes]
            np.round((stride * inds[0] + 1.0 + cell_size) / scale), # 形状 ： [n_boxes]
            score,  # 形状 ： [n_boxes]
            offsets,  # 形状 ： [4, n_boxes]
        ]
    )
    # why one is added?

    # 每个边界框的格式为：
    # 1. xmin ：边界框左上角 x 坐标
    # 2. ymin ：边界框左上角 y 坐标
    # 3. xmax ：边界框右下角 x 坐标
    # 4. ymax ：边界框右下角 y 坐标
    # 5. score ：人脸置信度
    # 6. tx1 ：左上角 x 坐标的偏移量
    # 7. ty1 ：左上角 y 坐标的偏移量
    # 8. tx2 ：右下角 x 坐标的偏移量
    # 9. ty2 ：右下角 y 坐标的偏移量
    return bounding_boxes.T # 形状 ： [n_boxes, 9]
