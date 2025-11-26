import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
from tqdm import tqdm
from data.data import *
from torchvision import transforms
from torch.utils.data import DataLoader
from net.CATFormer import CATFormer
import time

eval_parser = argparse.ArgumentParser(description='Eval')
eval_parser.add_argument('--perc', action='store_true', help='trained with perceptual loss')
eval_parser.add_argument('--lol', action='store_true', help='output lolv1 dataset')
eval_parser.add_argument('--lol_v2_real', action='store_true', help='output lol_v2_real dataset')
eval_parser.add_argument('--lol_v2_syn', action='store_true', help='output lol_v2_syn dataset')

eval_parser.add_argument('--best_GT_mean', action='store_true', help='output lol_v2_real dataset best_GT_mean')
eval_parser.add_argument('--best_PSNR', action='store_true', help='output lol_v2_real dataset best_PSNR')
eval_parser.add_argument('--best_SSIM', action='store_true', help='output lol_v2_real dataset best_SSIM')

eval_parser.add_argument('--custome', action='store_true', help='output custome dataset')
eval_parser.add_argument('--custome_path', type=str, default='./YOLO')
eval_parser.add_argument('--alpha', type=float, default=1.0)
eval_parser.add_argument('--gamma', type=float, default=1.0)
eval_parser.add_argument('--unpaired_weights', type=str, default='./weights/LOLv2_syn/w_perc.pth')

ep = eval_parser.parse_args()


def eval(model, testing_data_loader, model_path, output_folder, norm_size=True, LOL=False, v2=False, unpaired=False,
         alpha=1.0, gamma=1.0):
    torch.set_grad_enabled(False)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    print('Pre-trained model is loaded.')
    model.eval()
    print('Evaluation:')

    if LOL:
        model.dpa_trans.gated = True
    elif v2:
        model.dpa_trans.gated2 = True
        model.dpa_trans.alpha = alpha

    for batch in tqdm(testing_data_loader):
        with torch.no_grad():
            if norm_size:
                input, name = batch[0], batch[1]
            else:
                input, name, h, w = batch[0], batch[1], batch[2], batch[3]

            input = input.cuda()
            res = model(input ** gamma)
            if isinstance(res, tuple):
                output = res[0]
            else:
                output = res

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        output = torch.clamp(output.cuda(), 0, 1).cuda()
        if not norm_size:
            output = output[:, :, :h, :w]

        output_img = transforms.ToPILImage()(output.squeeze(0))
        output_img.save(output_folder + name[0])
        torch.cuda.empty_cache()
    print('===> End evaluation')

    if LOL:
        model.dpa_trans.gated = False
    elif v2:
        model.dpa_trans.gated2 = False
    torch.set_grad_enabled(True)


if __name__ == '__main__':

    cuda = True
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, or need to change CUDA_VISIBLE_DEVICES number")

    if not os.path.exists('./output'):
        os.mkdir('./output')

    norm_size = True
    num_workers = 1
    alpha = None
    if ep.lol:
        eval_data = DataLoader(dataset=get_eval_set("../datasets/LOLv1"), num_workers=num_workers,
                               batch_size=1, shuffle=False)
        output_folder = './output/LOLv1/'
        if ep.perc:
            weight_path = './weights/LOLv1/w_perc.pth'
        else:
            weight_path = './weights/LOLv1/wo_perc.pth'


    elif ep.lol_v2_real:
        eval_data = DataLoader(dataset=get_eval_set("../datasets/LOLv2/Real_captured/Test/Low"),
                               num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/LOLv2_real/'
        if ep.best_GT_mean:
            weight_path = './weights/LOLv2_real/epoch_340.pth'
            alpha = 0.84
        elif ep.best_PSNR:
            weight_path = './weights/LOLv2_real/epoch_190.pth'
            alpha = 0.8
        elif ep.best_SSIM:
            weight_path = './weights/LOLv2_real/epoch_530.pth'
            alpha = 0.82

    elif ep.lol_v2_syn:
        eval_data = DataLoader(dataset=get_eval_set("./datasets/LOLv2/Synthetic/Test/Low"), num_workers=num_workers,
                               batch_size=1, shuffle=False)
        output_folder = './output/LOLv2_syn/'
        if ep.perc:
            weight_path = './weights/LOLv2_syn/w_perc.pth'
        else:
            weight_path = './weights/LOLv2_syn/wo_perc.pth'

    eval_net = CATFormer().cuda()
    eval(eval_net, eval_data, weight_path, output_folder, norm_size=norm_size, LOL=ep.lol, v2=ep.lol_v2_real,
         unpaired=ep.unpaired, alpha=alpha, gamma=ep.gamma)