import os
import time
import argparse

import pandas
from tqdm import tqdm
import torchvision.transforms.functional as TF

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from dataloader.crackDatasets import crackDataset_withname
from models.builder import EncoderDecoder as segmodel

from utils import metrics_crack
from utils.loss_crack import SoftDiceLoss
from engine.engine import Engine
from engine.logger import get_logger
from tensorboardX import SummaryWriter


os.environ['MASTER_PORT'] = '16005'

def test(config,modelpath,threshold=0.5):
    logger = get_logger()
    parser = argparse.ArgumentParser()
    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()

        print("=======================================")
        print(config.tb_dir)
        print("=======================================")

        cudnn.benchmark = True
        seed = config.seed
        if engine.distributed:
            seed = engine.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        if config.dataset_name == 'Deepcrack':
            root = "./data/Deepcrack/"
        elif config.dataset_name == 'crack260':
            root = "./data/crack260/"
        elif config.dataset_name == 'CFD':
            root = "./data/CrackForest/"
        elif config.dataset_name == 'Crack315':
            root = "./data/Crack315/"

        saveroot="/home/DBCNet/testresult/"
        if not os.path.exists(saveroot):
            os.mkdir(saveroot)
        savepath=saveroot+"0"
        while(os.path.exists(savepath)):
            savepath=savepath[:-1]+str(int(savepath[-1])+1)
        os.mkdir(savepath)

        describepath = os.path.join(savepath, config.backbone+"&"+config.decoder+".txt")
        with open(describepath, "w") as f:
            f.write("en=" + config.backbone + "\n" + "de=" + config.decoder)
            f.close()
        batchsize = config.batch_size
        outnum = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        val_dataset = crackDataset_withname(root, txt="test.txt")
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                 shuffle=False, num_workers=4, drop_last=True)

        if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
            tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
            generate_tb_dir = config.tb_dir + '/tb'
            tb = SummaryWriter(log_dir=tb_dir)
            engine.link_tb(tb_dir, generate_tb_dir)

        criterion = SoftDiceLoss()

        if engine.distributed:
            BatchNorm2d = nn.SyncBatchNorm
        else:
            BatchNorm2d = nn.BatchNorm2d

        model = segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)

        checkpoint = torch.load(modelpath)
        new_state_dict = {}
        # 遍历原始参数字典，修改键值并添加到新的参数字典中
        for key, value in checkpoint['model'].items():
            # 重命名键值，使其与模型定义的键值匹配
            if key.startswith('module.'):
                new_key = key[7:]  # 移除 'module.' 前缀
            else:
                new_key = key
            new_state_dict[new_key] = value

        # 使用新的参数字典加载模型
        model.load_state_dict(new_state_dict, strict=True)
        model.to(device)

        bar2 = tqdm(enumerate(val_loader), total=len(val_loader))
        bar2.set_description('--- eval --- :')
        model.eval()
        with torch.no_grad():
            loss = 0
            acc = 0
            precision = 0
            recall = 0
            f1 = 0
            iou = 0
            miou = 0
            for idx, (img, label, imgpath) in bar2:
                img = img.to(device)
                label = label.to(device)
                if outnum == 5:
                    pred_output, out1, out2, out3, out4, out5 = model(img)
                    val_loss = criterion(pred_output.view(-1, 1), label.view(-1, 1))
                elif outnum == 2:
                    pred_output, pout1 = model(img)
                    val_loss = criterion(pred_output.view(-1, 1), label.view(-1, 1)) + criterion(pout1.view(-1, 1),
                                                                                                 label.view(-1, 1))

                elif outnum == 1:
                    pred_output = model(img)
                    val_loss = criterion(pred_output.view(-1, 1), label.view(-1, 1))

                loss += val_loss

                pred = torch.sigmoid(pred_output)
                # 将输出转换为二值图像
                # threshold = 0.5
                binary_pred = (pred[0] >= threshold).float()

                # 将二值图像转换为PIL图像
                binary_pred_pil = TF.to_pil_image(binary_pred)
                imgname = imgpath[0].split('/')[-1]
                pred_savepath = os.path.join(savepath, "pred")
                if not os.path.exists(pred_savepath):
                    os.mkdir(pred_savepath)
                # 保存二值图像
                binary_pred_pil.save(os.path.join(pred_savepath, imgname[:-4] + ".png"))
                # pred = pred_output
                ac, p, r, f, = metrics_crack.f1_loss(label[0], pred,thresh=threshold)
                acc += ac
                precision += p
                recall += r
                f1 += f
                i, _ = metrics_crack.iou_score(pred, label[0],thresh=threshold)
                iou += i
                mi = metrics_crack.miou(pred, label[0],thresh=threshold)
                miou += mi
                acc_per_img = [imgname, ac, p, r, f, i, mi]
                data_valloss = pandas.DataFrame([acc_per_img])
                data_valloss.to_csv(savepath + '/acc_per_img.csv', mode='a', header=False, index=False)
            l = len(val_loader)
            acclist = [acc / l * 100, precision / l * 100, recall / l * 100, f1 / l * 100, iou / l * 100,
                       miou / l * 100, loss.item() / l * 100]
            vallosslist = [loss.item() / len(val_loader) * 100]
            print("valloss", vallosslist)
            data_valloss = pandas.DataFrame([vallosslist])
            data_valloss.to_csv(savepath + '/valloss.csv', mode='a', header=False, index=False)
            print("acc", acclist)
            data_acc = pandas.DataFrame([acclist])
            data_acc.to_csv(savepath + '/acc.csv', mode='a', header=False, index=False)

if __name__=="__main__":
    from configs.config_crack_Deepcrack import config
    for threshold in [50]:
        print("threshold", threshold / 100)
        filepath="/home/Deepcrack8784"
        describefile=os.path.join(filepath,"describe.txt")
        with open(describefile,'r') as f:
            de=f.readlines()
            config.backbone = de[0][3:-1]
            config.decoder = de[1][3:-1]
            print(config.decoder)
            modelfile=os.path.join(filepath,"checkpoint")
            for name in os.listdir(modelfile):
                if name.endswith(".pth"):
                    modelpath=os.path.join(modelfile,name)
            test(config,modelpath=modelpath,threshold=threshold / 100)