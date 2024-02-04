import torch.nn
import argparse
import os
import math
import utils
from model.ARWGAN import *
from noise_argparser import NoiseArgParser
from noise_layers.noiser import Noiser
from PIL import Image
import torchvision.transforms.functional as TF
import pandas as pd
from secretsharing import secret_int_to_points, points_to_secret_int

def produce_points(secret, n, k):
    """共享秘密生成"""
    points = secret_int_to_points(secret, n, k)
    list_points = []
    for i in points:
        bins_serial = bin(i[0])[2:]
        bins_secrets = bin(i[1])[2:]
        # 将子秘密转换为长度为30的二进制列表，前20位为秘密，后10位为序号
        if len(bins_secrets) <= 20:
            bins_secrets = '0' * (20 - len(bins_secrets)) + bins_secrets
        if len(bins_serial) <= 10:
            bins_serial = '0' * (10 - len(bins_serial)) + bins_serial
        l = list(bins_secrets + bins_serial)
        L = [float(i) for i in l]
        list_points.append(L)
    return list_points

def restruct(list_points):
    """秘密重构"""
    points = []
    for i in list_points:
        bins_secrets = ''.join([str(int(i)) for i in i[:20]])
        bins_serial = ''.join([str(int(i)) for i in i[20:]])
        points.append((int(bins_serial, 2), int(bins_secrets, 2)))
    return points_to_secret_int(points)

def secret_gen(d :int):
    """秘密生成"""
    secret = [0]*d
    import random
    for i in range(d):
        bit_string = ''.join(random.choice(['0', '1']) for _ in range(18))
        integer_value = int(bit_string, 2)
        secret[i] = integer_value
    return secret

def secret_share(secret, n, k):
    """秘密共享"""
    p = []
    for i in range(len(secret)):
        p.append(produce_points(secret[i], n, k))
    return p

def randomCrop(img, height, width):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = np.random.randint(0, img.shape[1] - width)
    y = np.random.randint(0, img.shape[0] - height)
    img = img[y:y + height, x:x + width]
    return img


def PSNR(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def yuv_psnr(img):
    imgy = 0.299 * img[:, 0, :, :] + 0.587 * img[:, 1, :, :] + 0.114 * img[:, 2:, :, :]
    imgu = -0.14713 * img[:, 0, :, :] + (-0.28886) * img[:, 1, :, :] + 0.436 * img[:, 2:, :, :]
    imgv = 0.615 * img[:, 0, :, :] + -0.51499 * img[:, 1, :, :] + (-0.10001) * img[:, 2:, :, :]
    return imgy, imgu, imgv

def most_common_bits(bit_lists):
    if not bit_lists:
        return []

    list_length = len(bit_lists[0])
    bit_counts = [{'0': 0, '1': 0} for _ in range(list_length)]

    # 统计每个位置的比特
    for bit_list in bit_lists:
        for i, bit in enumerate(bit_list):
            bit_str = '1' if bit == 1.0 else '0'
            bit_counts[i][bit_str] += 1

    # 确定每个位置上出现次数最多的比特
    most_common = [max(count, key=count.get) for count in bit_counts]
    return [int(bit) for bit in most_common]


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    parser = argparse.ArgumentParser(description='Test trained models')
    parser.add_argument('--options-file', '-o', default='options-and-config.pickle', type=str,
                        help='The file where the simulation options are stored.')
    parser.add_argument('--checkpoint-file', '-c', required=True, type=str, help='Model checkpoint file')
    parser.add_argument('--batch-size', '-b', default=12, type=int, help='The batch size.')
    parser.add_argument('--source_images', '-s', required=True, type=str,
                        help='The image to watermark')
    parser.add_argument("--noise", '-n', nargs="*", action=NoiseArgParser)
    # parser.add_argument('--times', '-t', default=10, type=int,
    #                     help='Number iterations (insert watermark->extract).')

    args = parser.parse_args()
    train_options, net_config, noise_config = utils.load_options(args.options_file)
    noise_config = args.noise
    noiser = Noiser(noise_config, device)

    # 此处为修改行
    # checkpoint = torch.load(args.checkpoint_file)
    checkpoint = torch.load(args.checkpoint_file, map_location="cpu")
    hidden_net = ARWGAN(net_config, device, noiser, None)
    utils.model_from_checkpoint(hidden_net, checkpoint)
    source_images = os.listdir(args.source_images)

    #添加代码
    d = 1
    n = 5
    t = 4
    m = 10
    k = 20
    assert d*n*k <= 2**m-1, "d*n*k should be less than 2^m-1"
    secret = secret_gen(d)
    secret_s = secret_share(secret, t, n)
    print(secret_s)

    num = 0


    result_sd = [[[0 for _ in range(k)] for _ in range(n)] for _ in range(d)]
    for i in range(d):
        for j in range(n):
            for z in range(k):
                num += 1
                source_image = str(1)+'.png'
                new_values = secret_s[i][j]

                image_pil = Image.open(os.path.join(args.source_images, source_image))
                image_pil = image_pil.resize((net_config.H, net_config.W))
                image_tensor = TF.to_tensor(image_pil).to(device)
                image_tensor = image_tensor * 2 - 1
                image_tensor.unsqueeze_(0)
                # np.random.seed(42)

                message = torch.Tensor(np.random.choice([0, 1], (image_tensor.shape[0],
                                                                 net_config.message_length))).to(device)

                batch_size = message.shape[0]
                new_message = torch.tensor([new_values] * batch_size, dtype=torch.float32).to(device)
                message = new_message

                losses, (encoded_images, noised_images, decoded_messages) = hidden_net.validate_on_batch(
                    [image_tensor, message])

                decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
                message_detached = message.detach().cpu().numpy()

                # print(message_detached)
                # print(decoded_rounded)
                decoded_rounded[0][0] = 0
                result_sd[i][j][z] = decoded_rounded[0]
                print("还原数据：",decoded_rounded[0])
                print("原数据：",message_detached[0])
                print('error : {:.3f}'.format(np.mean(np.abs(decoded_rounded - message_detached))))
                utils.save_images(image_tensor.cpu(), encoded_images.cpu(), f'test{num}', './images',
                                  resize_to=(128, 128))
                # result_sd[i][j][z] = message_detached[0]
                # print(decoded_rounded[0])

    print(secret)
    for i in range(d):
        list = []
        for j in range(n):
            # print(secret_s[i][j])
            print(most_common_bits(result_sd[i][j]))
            list.append(most_common_bits(result_sd[i][j]))
        print(restruct(list))








    # for source_image in source_images:
    #     # 此处为修改行
    #     print(source_image)
    #     # image_pil = Image.open(args.source_images + source_image)
    #     image_pil = Image.open(os.path.join(args.source_images, source_image))
    #     image_pil = image_pil.resize((net_config.H, net_config.W))
    #     image_tensor = TF.to_tensor(image_pil).to(device)
    #     image_tensor = image_tensor * 2 - 1
    #     image_tensor.unsqueeze_(0)
    #     # np.random.seed(42)
    #
    #     message = torch.Tensor(np.random.choice([0, 1], (image_tensor.shape[0],
    #                                                      net_config.message_length))).to(device)
    #     print(net_config.message_length)
    #
    #     losses, (encoded_images, noised_images, decoded_messages) = hidden_net.validate_on_batch(
    #         [image_tensor, message])
    #
    #     decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
    #     message_detached = message.detach().cpu().numpy()
    #     print(message_detached)
    #     print(decoded_rounded)
    #     print('original: {}'.format(message_detached))
    #     print('decoded : {}'.format(decoded_rounded))
    #     print('error : {:.3f}'.format(np.mean(np.abs(decoded_rounded - message_detached))))

        # # 添加代码
        # utils.save_images(image_tensor.cpu(), encoded_images.cpu(), f'test{num}', './images', resize_to=(128, 128))


    # data_df = pd.DataFrame({'error': excel_error, 'PSNR': excel_psnr})
    # data_df.to_csv('error_rate.csv', index=False, sep=',')

    # 修改行
    # utils.save_images(image_tensor.cpu(), encoded_images.cpu(), 'test', './images', resize_to=(128, 128))


if __name__ == '__main__':
    main()
