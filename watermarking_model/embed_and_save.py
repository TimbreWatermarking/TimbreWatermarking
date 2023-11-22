import os
import torch
import yaml
import logging
import argparse
import warnings
import numpy as np
from rich.progress import track
from torch.utils.data import DataLoader
from model.loss import Loss
from torch.nn.functional import mse_loss
import soundfile
import random
import pdb


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
    train_config["path"]["wm_speech"] = args.save_path
    train_config["path"]["raw_path_test"] = args.original_path
    model_config["test"]["model_path"] = args.model_path

    pre_step = 0
    if model_config["structure"]["transformer"]:
        if model_config["structure"]["mel"]:
            from model.mel_modules import Encoder, Decoder
            from dataset.data import mel_dataset as my_dataset
        else:
            from model.modules import Encoder, Decoder
            from dataset.data import twod_dataset as my_dataset
    elif model_config["structure"]["conv2"]:
        from model.conv2_modules import Encoder, Decoder, Discriminator
        from dataset.data import mel_dataset_test as my_dataset
    elif model_config["structure"]["conv2mel"]:
        if not model_config["structure"]["ab"]:
            logging.info("use conv2mel model")
            from model.conv2_mel_modules import Encoder, Decoder, Discriminator
            from dataset.data import mel_dataset_test as my_dataset
        else:
            logging.info("use ablation conv2mel model")
            from model.conv2_mel_modules_ab import Encoder, Decoder, Discriminator
            from dataset.data import mel_dataset_test as my_dataset
    else:
        from model.conv_modules import Encoder, Decoder
        from dataset.data import oned_dataset as my_dataset
    # ---------------- get train dataset
    audios = my_dataset(process_config=process_config, train_config=train_config)
    batch_size = 1
    assert batch_size < len(audios)
    audios_loader = DataLoader(audios, batch_size=batch_size, shuffle=False)

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
    # encoder = model["encoder"]
    # decoder = model["decoder"]
    encoder.load_state_dict(model["encoder"])
    decoder.load_state_dict(model["decoder"], strict=False)
    encoder.eval()
    decoder.eval()
    decoder.robust = False
    # ---------------- Loss
    loss = Loss(train_config=train_config)

    # ---------------- Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)

    # ---------------- Embedding
    logging.info(logging_mark + "\t" + "Begin Embedding" + "\t" + logging_mark)
    epoch_num = train_config["iter"]["epoch"]
    save_circle = train_config["iter"]["save_circle"]
    show_circle = train_config["iter"]["show_circle"]
    lambda_e = train_config["optimize"]["lambda_e"]
    lambda_m = train_config["optimize"]["lambda_m"]
    global_step = 0
    
    if not model_config["structure"]["ab"]:
        wm_path = os.path.join(train_config["path"]["wm_speech"],"wmed-"+str(args.wm),"wavs")
    else:
        wm_path = os.path.join(train_config["path"]["wm_speech"],"wmed_ab-"+str(args.wm),"wavs")
        

    ref_path = os.path.join(train_config["path"]["wm_speech"],"ref")
    if not os.path.exists(wm_path): os.makedirs(wm_path)
    if not os.path.exists(ref_path): os.makedirs(ref_path)
    train_len = len(audios_loader)

    wm_list = []
    with torch.no_grad():
        for sample in track(audios_loader):
            if global_step > 10: break
            global_step += 1
            # ---------------- build watermark
            wmp = open("results/wmpool.txt", 'r')
            wmplist = wmp.readlines()
            wmp.close()
            wm = eval(wmplist[args.wm])
            msg = np.array([[wm]])
            
            wm_list.append(msg)
            msg = torch.from_numpy(msg).float()*2 - 1
            wav_matrix = sample["matrix"].to(device)
            sample_rate = sample["sample_rate"]
            msg = msg.to(device)
            # pdb.set_trace()
            encoded, carrier_wateramrked = encoder.test_forward(wav_matrix, msg)
            name = sample["name"][0]
            soundfile.write(os.path.join(wm_path, name), encoded.cpu().squeeze(0).squeeze(0).detach().numpy(), samplerate=sample_rate)
            # soundfile.write(os.path.join(ref_path, name), wav_matrix.cpu().squeeze(0).squeeze(0).detach().numpy(), samplerate=sample_rate)
            
            decoded = decoder.test_forward(encoded)
            losses = loss.en_de_loss(wav_matrix, encoded, msg, decoded)
            decoder_acc = (decoded >= 0).eq(msg >= 0).sum().float() / msg.numel()
            zero_tensor = torch.zeros(wav_matrix.shape).to(device)
            snr = 10 * torch.log10(mse_loss(wav_matrix.detach(), zero_tensor) / mse_loss(wav_matrix.detach(), encoded.detach()))
            norm2=mse_loss(wav_matrix.detach(),zero_tensor)
            logging.info('-' * 100)
            logging.info("step:{} - wav_loss:{:.8f} - msg_loss:{:.8f} - acc:{:.8f} - snr:{:.8f} - norm:{:.8f} - patch_num:{} - pad_num:{} - wav_len:{} - name:{}".format( \
                global_step, losses[0], losses[1], decoder_acc, snr, norm2, sample["patch_num"].item(), sample["pad_num"].item(), wav_matrix.shape[2], sample["name"][0]))
        np.save("results/wm_speech/wm.npy", np.stack(wm_list,axis=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--process_config",
        type=str,
        default="config/process.yaml",
        help="path to process.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, default="config/model.yaml", help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, default="config/train.yaml", help="path to train.yaml"
    )
    parser.add_argument("--wm", type=int, default=0, help="Index of the watermark in results/wmpool.txt")
    parser.add_argument("-o", "--original_path", type=str, default="data/ljspeech/LJSpeech-1.1/wavs/", help="original wavs path")
    parser.add_argument("-s", "--save_path", type=str, default="results/wm_speech/ljspeech", help="path to save watermarked wavs")
    parser.add_argument("-mp", "--model_path", type=str, default="results/ckpt/pth/", help="model ckpt.pth.tar path")
    args = parser.parse_args()

    # Read Config
    process_config = yaml.load(
        open(args.process_config, "r"), Loader=yaml.FullLoader
    )
    
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (process_config, model_config, train_config)

    main(args, configs)
