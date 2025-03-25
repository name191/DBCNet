import os.path as osp
import os
import sys
import time
import argparse
import pandas
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel
from dataloader.crackDatasets import crackDataset
from models.builder import EncoderDecoder as segmodel
from utils import metrics_crack
from utils.init_func import group_weight
from utils.loss_crack import SoftDiceLoss
from utils.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from utils.pyt_utils import all_reduce_tensor
from engine.logger import get_logger
from tensorboardX import SummaryWriter


os.environ['MASTER_PORT'] = '16005'

def main(config):
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

        batchsize = config.batch_size
        outnum = 1

        trainset = crackDataset(root, txt="train.txt")
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                                   shuffle=True, num_workers=4, drop_last=True)

        val_dataset = crackDataset(root, txt="test.txt")
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

        # group weight and config optimizer
        base_lr = config.lr
        if engine.distributed:
            base_lr = config.lr

        params_list = []
        params_list = group_weight(params_list, model, BatchNorm2d, base_lr)

        if config.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
        elif config.optimizer == 'SGDM':
            optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
        else:
            raise NotImplementedError

        # config lr policy
        total_iteration = config.nepochs * config.niters_per_epoch
        lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

        if engine.distributed:
            logger.info('.............distributed training.............')
            if torch.cuda.is_available():
                model.cuda()
                model = DistributedDataParallel(model, device_ids=[engine.local_rank],
                                                output_device=engine.local_rank, find_unused_parameters=False)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

        engine.register_state(dataloader=train_loader, model=model,
                              optimizer=optimizer)
        if engine.continue_state_object:
            engine.restore_checkpoint()

        optimizer.zero_grad()
        model.train()
        logger.info('begin trainning:')

        best_mean_iou = 0.0  # Track the best mean IoU for model saving
        best_epoch = 100000  # Track the epoch with the best mean IoU for model saving

        describepath = os.path.join(config.log_dir, "describe.txt")
        with open(describepath, "w") as f:
            f.write("en=" + config.backbone + "\n" + "de=" + config.decoder+ "\n" + "lr=" + str(config.lr)+ "\n" + "epoch=" + str(config.nepochs))
            f.close()
        for epoch in range(engine.state.epoch, config.real_nepochs + 1):
            bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
            pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                        bar_format=bar_format)
            dataloader = iter(train_loader)

            sum_loss = 0

            for idx in pbar:
                engine.update_iteration(epoch, idx)

                (imgs, gts) = next(dataloader)

                imgs = imgs.cuda(non_blocking=True)
                gts = gts.cuda(non_blocking=True)

                loss = model(imgs, gts)

                if engine.distributed:
                    reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                current_idx = (epoch - 1) * config.niters_per_epoch + idx
                lr = lr_policy.get_lr(current_idx)

                for i in range(len(optimizer.param_groups)):
                    optimizer.param_groups[i]['lr'] = lr

                if engine.distributed:
                    if dist.get_rank() == 0:
                        sum_loss += reduce_loss.item()
                        print_str = 'Epoch {}/{}'.format(epoch, config.real_nepochs) \
                                    + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                                    + ' lr=%.4e' % lr \
                                    + ' loss=%.4f total_loss=%.4f' % (reduce_loss.item(), (sum_loss / (idx + 1)))
                        pbar.set_description(print_str, refresh=False)
                else:
                    sum_loss += loss
                    print_str = 'Epoch {}/{}'.format(epoch, config.real_nepochs) \
                                + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                                + ' lr=%.4e' % lr \
                                + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))
                    pbar.set_description(print_str, refresh=False)
                del loss

            if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
                tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)

            if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (
                    epoch == config.nepochs):
                if engine.distributed and (engine.local_rank == 0):
                    engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                    config.log_dir,
                                                    config.log_dir_link)
                elif not engine.distributed:
                    engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                    config.log_dir,
                                                    config.log_dir_link)

            torch.cuda.empty_cache()
            if (epoch >= config.checkpoint_start_epoch) and (
                    epoch - config.checkpoint_start_epoch) % config.checkpoint_step == 0:
                bar2 = tqdm(enumerate(val_loader), total=len(val_loader))
                bar2.set_description('Epoch %d --- eval --- :' % epoch)
                model.eval()
                with torch.no_grad():
                    loss = 0
                    acc = 0
                    precision = 0
                    recall = 0
                    f1 = 0
                    iou = 0
                    miou = 0
                    for idx, (img, label) in bar2:
                        img = img.to(device)
                        label = label.to(device)
                        if outnum == 5:
                            pred_output, out1, out2, out3, out4, out5 = model(img)
                            val_loss = criterion(pred_output.view(-1, 1), label.view(-1, 1)) / batchsize
                        elif outnum == 2:
                            pred_output, pout1 = model(img)
                            val_loss = criterion(pred_output.view(-1, 1), label.view(-1, 1)) + criterion(
                                pout1.view(-1, 1),
                                label.view(-1, 1))

                        elif outnum == 1:
                            pred_output = model(img)
                            val_loss = criterion(pred_output.view(-1, 1), label.view(-1, 1)) / batchsize

                        loss += val_loss

                        pred = torch.sigmoid(pred_output)
                        # pred = pred_output
                        ac, p, r, f, = metrics_crack.f1_loss(label[0], pred)
                        acc += ac
                        precision += p
                        recall += r
                        f1 += f
                        i, _ = metrics_crack.iou_score(pred, label[0])
                        iou += i
                        mi = metrics_crack.miou(pred, label[0])
                        miou += mi
                    # [acc,precision,recall,f1,mIoU]
                    l = len(val_loader)
                    acclist = [acc / l * 100, precision / l * 100, recall / l * 100, f1 / l * 100, iou / l * 100,
                               miou / l * 100, loss.item() / l * 100, epoch]
                    vallosslist = [loss.item() / len(val_loader) * 100]
                    print("valloss", vallosslist)
                    data_valloss = pandas.DataFrame([vallosslist])
                    data_valloss.to_csv(config.log_dir + '/valloss.csv', mode='a', header=False, index=False)
                    print("acc", acclist)
                    data_acc = pandas.DataFrame([acclist])
                    data_acc.to_csv(config.log_dir + '/acc.csv', mode='a', header=False, index=False)
                    all_acc=pandas.read_csv(config.log_dir + '/acc.csv',header=None)
                    sort_all_acc=all_acc.sort_values(by=all_acc.columns[-3],ascending=False)
                    sort_all_acc.to_csv(config.log_dir + '/acc_sort.csv',header=False, index=False)

                    mean_IoU = miou / l * 100
                    print('mean_IoU:', mean_IoU)

                    # Determine if the model performance improved
                    if mean_IoU > best_mean_iou:
                        # If the model improves, remove the saved checkpoint for this epoch
                        checkpoint_path = os.path.join(config.checkpoint_dir, f'epoch-{best_epoch}.pth')
                        if os.path.exists(checkpoint_path):
                            os.remove(checkpoint_path)
                        best_epoch = epoch
                        best_mean_iou = mean_IoU
                    else:
                        # If the model does not improve, remove the saved checkpoint for this epoch
                        checkpoint_path = os.path.join(config.checkpoint_dir, f'epoch-{epoch}.pth')
                        if os.path.exists(checkpoint_path):
                            os.remove(checkpoint_path)
                model.train()

if __name__=="__main__":
    from configs.config_crack_Deepcrack import config
    for dename in ["myMambaDecoder"]:
        for enname in ["my_backbone"]:
            while (os.path.exists(config.log_dir)):
                config.log_dir = config.log_dir[:-1] + str(int(config.log_dir[-1]) + 1)
            config.tb_dir = osp.abspath(osp.join(config.log_dir, "tb"))
            config.log_dir_link = config.log_dir
            config.checkpoint_dir = osp.abspath(osp.join(config.log_dir, "checkpoint"))
            config.backbone = enname
            config.decoder = dename

            main(config)

