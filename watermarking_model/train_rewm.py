import os
import torch
import yaml
import logging
import argparse
import warnings
import numpy as np
from torch.optim import Adam
from rich.progress import track
from torch.utils.data import DataLoader
from model.loss import Loss, Loss_identity
from utils.tools import save, log, save_op
from utils.optimizer import ScheduledOptimMain, ScheduledOptimDisc, my_step
from itertools import chain
from torch.nn.functional import mse_loss
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import random
import pdb
import torch.nn as nn
import shutil



# set seeds
seed = 2022
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)



logging_mark = "#"*20
# warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args, configs):
    logging.info('main function')
    process_config, model_config, train_config = configs
    train_config["path"]["ckpt"] = "results/ckpt_instruct_rewm/"
    os.makedirs(train_config["path"]["ckpt"], exist_ok=True)
    train_config["watermark"]["length"] = 10
    train_config["iter"]["epoch"] = 20
    model_config["conv2"]["block"] = 'skip'
    train_config["iter"]["save_circle"] = 2
    add_loss = nn.MSELoss()
    train_config["optimize"]["lambda_e"] = 1
    train_config["optimize"]["lambda_m"] = 0.01
    logging.info("lambda_e:{}\nlambda_m:{}\n".format(train_config["optimize"]["lambda_e"], train_config["optimize"]["lambda_m"]))
    model_config["test"]["model_path"] = "results/ckpt/pth/" 
    model_config["test"]["index"] = -2


    pre_step = 0
    if model_config["structure"]["transformer"]:
        if model_config["structure"]["mel"]:
            from model.mel_modules import Encoder, Decoder
            from dataset.data import mel_dataset as my_dataset
        else:
            from model.modules import Encoder, Decoder
            from dataset.data import twod_dataset as my_dataset
    elif model_config["structure"]["conv2mel"]:
        if not model_config["structure"]["ab"]:
            logging.info("use conv2mel model")
            from model.conv2_mel_rewm_modules import Encoder, Decoder, Discriminator
            from dataset.data import wav_dataset_wopreload as my_dataset
        else:
            logging.info("use ablation conv2mel model")
            from model.conv2_mel_modules_ab import Encoder, Decoder, Discriminator
            from dataset.data import wav_dataset as my_dataset
    # ---------------- get train dataset
    audios = my_dataset(process_config=process_config, train_config=train_config, flag='train')
    val_audios = my_dataset(process_config=process_config, train_config=train_config, flag='val')

    batch_size = train_config["optimize"]["batch_size"]
    assert batch_size < len(audios)

    audios_loader = DataLoader(audios, batch_size=batch_size, shuffle=True)
    val_audios_loader = DataLoader(val_audios, batch_size=batch_size, shuffle=False)
    # ---------------- build model
    win_dim = process_config["audio"]["win_len"]
    embedding_dim = model_config["dim"]["embedding"]
    nlayers_encoder = model_config["layer"]["nlayers_encoder"]
    nlayers_decoder = model_config["layer"]["nlayers_decoder"]
    attention_heads_encoder = model_config["layer"]["attention_heads_encoder"]
    attention_heads_decoder = model_config["layer"]["attention_heads_decoder"]
    msg_length = train_config["watermark"]["length"]
    if model_config["structure"]["mel"] or model_config["structure"]["conv2"]:
        encoder = Encoder(process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_encoder=nlayers_encoder, attention_heads=attention_heads_encoder).to(device)
        decoder = Decoder(process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_decoder=nlayers_decoder, attention_heads=attention_heads_decoder).to(device)
    else:
        encoder = Encoder(model_config, msg_length, win_dim, embedding_dim, nlayers_encoder=nlayers_encoder, attention_heads=attention_heads_encoder).to(device)
        decoder = Decoder(model_config, msg_length, win_dim, embedding_dim, nlayers_decoder=nlayers_decoder, attention_heads=attention_heads_decoder).to(device)
    
    path_model = model_config["test"]["model_path"]
    model_name = model_config["test"]["model_name"]
    if model_name:
        model = torch.load(os.path.join(path_model, model_name))
    else:
        index = model_config["test"]["index"]
        model_list = os.listdir(path_model)
        model_list = sorted(model_list,key=lambda x:os.path.getmtime(os.path.join(path_model,x)))
        model_path = os.path.join(path_model, model_list[index])
        logging.info(model_path)
        model = torch.load(model_path)
        logging.info("model <<{}>> loadded".format(model_path))

    encoder.load_state_dict(model["encoder"])
    decoder.load_state_dict(model["decoder"], strict=False)
    
    
    # adv
    if train_config["adv"]:
        discriminator = Discriminator(process_config).to(device)
        # d_op = ScheduledOptimDisc(discriminator,train_config)
        d_op = Adam(
            params=chain(discriminator.parameters()),
            betas=train_config["optimize"]["betas"],
            eps=train_config["optimize"]["eps"],
            weight_decay=train_config["optimize"]["weight_decay"],
            lr = train_config["optimize"]["lr"]
        )
        lr_sched_d = StepLR(d_op, step_size=train_config["optimize"]["step_size"], gamma=train_config["optimize"]["gamma"])
    # shared parameters
    if model_config["structure"]["share"]:
        if model_config["structure"]["transformer"]:
            decoder.msg_decoder = encoder.encoder
        else:
            decoder.wav_encoder = encoder.wav_encoder
    # ---------------- optimizer
    # en_de_op = Adam([
    #     {'params': decoder.parameters(), 'lr': args.lr_en_de}, 
    #     {'params': encoder.parameters(), 'lr': args.lr_en_de}
	# ])
    # en_de_op = ScheduledOptimMain(encoder, decoder, train_config, model_config, args.restore_step)
    en_de_op = Adam(
            params=chain(decoder.parameters(), encoder.parameters()),
            betas=train_config["optimize"]["betas"],
            eps=train_config["optimize"]["eps"],
            weight_decay=train_config["optimize"]["weight_decay"],
            lr = train_config["optimize"]["lr"]
        )
    lr_sched = StepLR(en_de_op, step_size=train_config["optimize"]["step_size"], gamma=train_config["optimize"]["gamma"])

    # ---------------- Loss
    # loss = Loss(train_config=train_config)
    loss = Loss_identity(train_config=train_config)

    # ---------------- Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    # train_logger = SummaryWriter(train_log_path)
    # val_logger = SummaryWriter(val_log_path)

    # ---------------- train
    logging.info(logging_mark + "\t" + "Begin Trainging" + "\t" + logging_mark)
    epoch_num = train_config["iter"]["epoch"]
    save_circle = train_config["iter"]["save_circle"]
    show_circle = train_config["iter"]["show_circle"]
    lambda_e = train_config["optimize"]["lambda_e"]
    lambda_m = train_config["optimize"]["lambda_m"]
    global_step = 0
    train_len = len(audios_loader)
    for ep in range(1, epoch_num+1):
        encoder.train()
        decoder.train()
        discriminator.train()
        step = 0
        logging.info('Epoch {}/{}'.format(ep, epoch_num))
        
        for sample in track(audios_loader):
            global_step += 1
            step += 1
            # ---------------- build watermark
            msg = np.random.choice([0,1], [batch_size, 1, msg_length])
            msg = torch.from_numpy(msg).float()*2 - 1
            
            ################################################################ wm
            wav_matrix = sample["matrix"].to(device)
            msg = msg.to(device)
            weight = decoder.get_weight(wav_matrix)
            encoded, carrier_wateramrked = encoder(wav_matrix, msg, weight, global_step)
            decoded = decoder(encoded, global_step)
            losses = loss.en_de_loss(wav_matrix, encoded, msg, decoded)
            ################################################################ re-wm
            msg_rewm = np.random.choice([0,1], [batch_size, 1, msg_length])
            msg_rewm = torch.from_numpy(msg_rewm).float()*2 - 1
            msg_rewm = msg_rewm.to(device)
            weight2 = decoder.get_weight(encoded)
            encoded2, carrier_wateramrked = encoder(encoded, msg_rewm, weight2, global_step)
            decoded3 = decoder(encoded2, global_step)
            loss_rewm = add_loss(msg, decoded3[0]) + add_loss(msg, decoded3[1])

            if global_step < pre_step:
                sum_loss = lambda_m*losses[1]
            else:
                sum_loss = lambda_e*losses[0] + lambda_m*losses[1]
            
            # adv
            if train_config["adv"]:
                lambda_a = lambda_m = train_config["optimize"]["lambda_a"]
                g_target_label_encoded = torch.full((batch_size, 1), 1, device=device).float()
                d_on_encoded_for_enc = discriminator(encoded)
                g_loss_adv = F.binary_cross_entropy_with_logits(d_on_encoded_for_enc, g_target_label_encoded)
                sum_loss += lambda_a*g_loss_adv
            
            sum_loss += loss_rewm * lambda_m

            sum_loss.backward()

            my_step(en_de_op, lr_sched, global_step, train_len)
            
            if train_config["adv"]:
                d_target_label_cover = torch.full((batch_size, 1), 1, device=device).float()
                d_on_cover = discriminator(wav_matrix)
                d_loss_on_cover = F.binary_cross_entropy_with_logits(d_on_cover, d_target_label_cover)
                d_loss_on_cover.backward()

                d_target_label_encoded = torch.full((batch_size, 1), 0, device=device).float()
                d_on_encoded = discriminator(encoded.detach())
                d_loss_on_encoded = F.binary_cross_entropy_with_logits(d_on_encoded, d_target_label_encoded)
                d_loss_on_encoded.backward()
                my_step(d_op, lr_sched_d, global_step, train_len)
            
            if step % show_circle == 0:
                # decoder_acc = (decoded[0] >= 0).eq(msg >= 0).sum().float() / msg.numel()
                decoder_acc = [((decoded[0] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item(), ((decoded[1] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item()]
                zero_tensor = torch.zeros(wav_matrix.shape).to(device)
                snr = 10 * torch.log10(mse_loss(wav_matrix.detach(), zero_tensor) / mse_loss(wav_matrix.detach(), encoded.detach()))
                norm2=mse_loss(wav_matrix.detach(),zero_tensor)
                logging.info('-' * 100)
                logging.info("step:{} - wav_loss:{:.8f} - msg_loss:{:.8f} - acc:[{:.8f},{:.8f}] - snr:{:.8f} - norm:{:.8f} - loss_rewm:{} - wav_len:{} - d_loss_on_encoded:{} - d_loss_on_cover:{}".format(\
                    step, losses[0], losses[1], decoder_acc[0], decoder_acc[1],\
                        snr, norm2, loss_rewm.item(), wav_matrix.shape[2], d_loss_on_encoded, d_loss_on_cover))
        
        # if ep % save_circle == 0 or ep == 1 or ep == 2:
        if ep % save_circle == 0:
            if not model_config["structure"]["ab"]:
                path = os.path.join(train_config["path"]["ckpt"], 'pth')
            else:
                path = os.path.join(train_config["path"]["ckpt"], 'pth_ab')
            save_op(path, ep, encoder, decoder, en_de_op)
            shutil.copyfile(os.path.realpath(__file__), os.path.join(path, os.path.basename(os.path.realpath(__file__)))) #save training scripts

        # varification
        with torch.no_grad():
            encoder.eval()
            decoder.eval()
            discriminator.eval()
            avg_acc = [0, 0]
            avg_snr = 0
            avg_wav_loss = 0
            avg_msg_loss = 0
            avg_d_loss_on_encoded = 0
            avg_d_loss_on_cover = 0
            avg_loss_rewm = 0
            # avg_loss_wm_free = 0
            count = 0
            for sample in track(val_audios_loader):
                count += 1
                # ---------------- build watermark
                msg = np.random.choice([0,1], [batch_size, 1, msg_length])
                msg = torch.from_numpy(msg).float()*2 - 1
                ################################################################ wm
                wav_matrix = sample["matrix"].to(device)
                msg = msg.to(device)
                weight = decoder.get_weight(wav_matrix)
                encoded, carrier_wateramrked = encoder(wav_matrix, msg, weight, global_step)
                decoded = decoder(encoded, global_step)
                losses = loss.en_de_loss(wav_matrix, encoded, msg, decoded)
                ################################################################ re-wm
                msg_rewm = np.random.choice([0,1], [batch_size, 1, msg_length])
                msg_rewm = torch.from_numpy(msg_rewm).float()*2 - 1
                msg_rewm = msg_rewm.to(device)
                weight2 = decoder.get_weight(encoded)
                encoded2, carrier_wateramrked = encoder(encoded, msg_rewm, weight2, global_step)
                decoded3 = decoder(encoded2, global_step)
                loss_rewm = add_loss(msg, decoded3[0]) + add_loss(msg, decoded3[1])

                # adv
                if train_config["adv"]:
                    lambda_a = lambda_m = train_config["optimize"]["lambda_a"]
                    g_target_label_encoded = torch.full((batch_size, 1), 1, device=device).float()
                    d_on_encoded_for_enc = discriminator(encoded)
                    g_loss_adv = F.binary_cross_entropy_with_logits(d_on_encoded_for_enc, g_target_label_encoded)
                if train_config["adv"]:
                    d_target_label_cover = torch.full((batch_size, 1), 1, device=device).float()
                    d_on_cover = discriminator(wav_matrix)
                    d_loss_on_cover = F.binary_cross_entropy_with_logits(d_on_cover, d_target_label_cover)

                    d_target_label_encoded = torch.full((batch_size, 1), 0, device=device).float()
                    d_on_encoded = discriminator(encoded.detach())
                    d_loss_on_encoded = F.binary_cross_entropy_with_logits(d_on_encoded, d_target_label_encoded)
                
                # decoder_acc = (decoded[0] >= 0).eq(msg >= 0).sum().float() / msg.numel()
                # decoder_acc = ((decoded[0] >= 0).eq(msg >= 0).sum().float() / msg.numel(), (decoded[1] >= 0).eq(msg >= 0).sum().float() / msg.numel())
                decoder_acc = [((decoded[0] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item(), ((decoded[1] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item()]
                zero_tensor = torch.zeros(wav_matrix.shape).to(device)
                snr = 10 * torch.log10(mse_loss(wav_matrix.detach(), zero_tensor) / mse_loss(wav_matrix.detach(), encoded.detach()))
                avg_acc[0] += decoder_acc[0]
                avg_acc[1] += decoder_acc[1]
                avg_snr += snr
                avg_wav_loss += losses[0]
                avg_msg_loss += losses[1]
                avg_d_loss_on_cover += d_loss_on_cover
                avg_d_loss_on_encoded += d_loss_on_encoded
                avg_loss_rewm += loss_rewm
            avg_acc[0] /= count
            avg_acc[1] /= count
            avg_snr /= count
            avg_wav_loss /= count
            avg_msg_loss /= count
            avg_d_loss_on_encoded /= count
            avg_d_loss_on_cover /= count
            avg_loss_rewm /= count
            logging.info('#e' * 60)
            logging.info("epoch:{} - wav_loss:{:.8f} - msg_loss:{:.8f} - loss_rewm:{} - acc:[{:.8f},{:.8f}] - snr:{:.8f} - d_loss_on_encoded:{} - d_loss_on_cover:{}".format(\
                ep, avg_wav_loss, avg_msg_loss, avg_loss_rewm.item(), avg_acc[0], avg_acc[1], avg_snr, avg_d_loss_on_encoded, avg_d_loss_on_cover))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--process_config",
        type=str,
        required=True,
        help="path to process.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    process_config = yaml.load(
        open(args.process_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (process_config, model_config, train_config)

    main(args, configs)
