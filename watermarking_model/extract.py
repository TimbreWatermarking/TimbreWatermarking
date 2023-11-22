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
from model.loss import Loss
import soundfile
import random
import pdb
import torchaudio

# set seeds
seed = 2022
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

logging_mark = "#"*20
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args, configs):
    logging.info('main function')
    process_config, model_config, train_config = configs
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
            from dataset.data import mel_dataset_test_2 as my_dataset
        else:
            logging.info("use ablation conv2mel model")
            from model.conv2_mel_modules_ab import Encoder, Decoder, Discriminator
            from dataset.data import mel_dataset_test as my_dataset
    else:
        from model.conv_modules import Encoder, Decoder
        from dataset.data import oned_dataset as my_dataset
        

    # for aim in aim_dirs:
    aim = args.target_path
    log = open("results/extract_log.out", "a+")
    train_config['path']['raw_path_test'] = aim
    # ---------------- get train dataset
    audios = my_dataset(process_config=process_config, train_config=train_config)
    batch_size = 1
    assert batch_size <= len(audios)
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
        # encoder = Encoder(process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_encoder=nlayers_encoder, attention_heads=attention_heads_encoder).to(device)
        decoder = Decoder(process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_decoder=nlayers_decoder, attention_heads=attention_heads_decoder).to(device)
    else:
        # encoder = Encoder(model_config, msg_length, win_dim, embedding_dim, nlayers_encoder=nlayers_encoder, attention_heads=attention_heads_encoder).to(device)
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
        # model = torch.load(model_path,map_location=torch.device('cpu'))
        model = torch.load(model_path)
        logging.info("model <<{}>> loadded".format(model_path))
        log.write("\n\nmodel <<{}>> loadded".format(model_path))
    # encoder = model["encoder"]
    # decoder = model["decoder"]
    # encoder.load_state_dict(model["encoder"])
    decoder.load_state_dict(model["decoder"],strict=False)
    # encoder.eval()
    decoder.eval()
    decoder.robust = False
    # ---------------- Loss
    loss = Loss(train_config=train_config)

    # ---------------- Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)

    # ---------------- Extracting
    logging.info(logging_mark + "\t" + "Begin Extracting" + "\t" + logging_mark)
    epoch_num = train_config["iter"]["epoch"]
    save_circle = train_config["iter"]["save_circle"]
    show_circle = train_config["iter"]["show_circle"]
    lambda_e = train_config["optimize"]["lambda_e"]
    lambda_m = train_config["optimize"]["lambda_m"]
    global_step = 0
    wm_path = os.path.join(train_config["path"]["wm_speech"],"wmed","wavs")
    ref_path = os.path.join(train_config["path"]["wm_speech"],"ref")
    if not os.path.exists(wm_path): os.makedirs(wm_path)
    if not os.path.exists(ref_path): os.makedirs(ref_path)
    train_len = len(audios_loader)

    wm_list = []
    avg_acc = 0

    with torch.no_grad():
        log.write("\ntest dir: {}\n".format(train_config["path"]["raw_path_test"]))
        logging.info("\ntest dir: {}\n".format(train_config["path"]["raw_path_test"]))
        wmp = open("results/wmpool.txt", 'r')
        wmplist = wmp.readlines()
        wmp.close()

        txtf = open("./ljs_audio_text_test_filelist.txt", 'r')
        txt = txtf.readlines()
        txtf.close()
        
        from utils.tools import fidelity
        avg_snr, avg_pesq, avg_utterance_sim, avg_spk_sim = 0, 0, 0, 0
        for sample in track(audios_loader):
            global_step += 1
            # wm_list.append(msg)
            wm = eval(wmplist[args.wm])
            msg = np.array([[wm]])
            msg = torch.from_numpy(msg).float()*2 - 1
            wav_matrix = sample["matrix"].to(device)
            sample_rate = sample["sample_rate"]
            msg = msg.to(device)
            # pdb.set_trace()
            # encoded, carrier_wateramrked = encoder(wav_matrix, msg)
            # encoded, carrier_wateramrked = encoder(wav_matrix, msg)
            name = sample["name"][0]
            try:
                this_ref_path = txt[int(name.split(".")[0])-1].split("|")[0]
                ref, ref_sr = torchaudio.load(this_ref_path)
            except Exception as e:
                print("cannot load ref audio of {}, set as random tensor insted".format(name))
                ref = torch.randn(wav_matrix.shape).squeeze(0)
                this_ref_path = "don't find"
            snr_score, pesq_score, utterance_sim_matrix, spk_sim_matrix = fidelity(ref.to(device).unsqueeze(0).detach(), wav_matrix.detach(), process_config["audio"]["sample_rate"])
            avg_pesq += pesq_score
            avg_snr += snr_score
            avg_spk_sim += spk_sim_matrix
            avg_utterance_sim += utterance_sim_matrix
            
            decoded = decoder.test_forward(wav_matrix)
            logging.info(decoded)
            decoder_acc = (decoded >= 0).eq(msg >= 0).sum().float() / msg.numel()
            zero_tensor = torch.zeros(wav_matrix.shape).to(device)
            losses = loss.en_de_loss(wav_matrix, zero_tensor, msg, decoded)
            norm2=losses[0]
            logging.info('-' * 100)
            logging.info("step:{} - snr:{:.8f} - pesq:{:.8f} - spk_sim:{:.8f} - utterance_sim:{:.8f} - acc:{:.8f} - msg_loss:{:.8f} - norm:{:.8f} - wav_len:{} - name:{} - refname:{}".format( \
                global_step, snr_score, pesq_score, spk_sim_matrix, utterance_sim_matrix, decoder_acc, losses[1], norm2, wav_matrix.shape[2], sample["name"][0], this_ref_path))

            log.write('-' * 100)
            log.write("\nstep:{} - snr:{:.8f} - pesq:{:.8f} - spk_sim:{:.8f} - utterance_sim:{:.8f} - acc:{:.8f} - msg_loss:{:.8f} - norm:{:.8f} - wav_len:{} - name:{} - refname:{}\n".format( \
                global_step, snr_score, pesq_score, spk_sim_matrix, utterance_sim_matrix, decoder_acc, losses[1], norm2, wav_matrix.shape[2], sample["name"][0], this_ref_path))
            avg_acc += decoder_acc
        logging.info("{} audios, avg_snr:{:.8f}, avg_pesq:{:.8f}, utterance_sim:{:.8f}, spk_sim:{:.8f}, avg_acc:{:.8f}".format(global_step, avg_snr/global_step, avg_pesq/global_step, avg_utterance_sim/global_step, avg_spk_sim/global_step, avg_acc/train_len))
        log.write("\n{} audios, avg_snr:{:.8f}, avg_pesq:{:.8f}, utterance_sim:{:.8f}, spk_sim:{:.8f}, avg_acc:{:.8f}".format(global_step, avg_snr/global_step, avg_pesq/global_step, avg_utterance_sim/global_step, avg_spk_sim/global_step, avg_acc/train_len))
        log.close()
        # np.save("results/wm_speech/wm.npy", np.stack(wm_list,axis=0))


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
    parser.add_argument(
        "--wm", type=int, default=0, help="Index of the watermark in results/wmpool.txt"
        )
    parser.add_argument(
        "-tp", "--target_path", type=str, default="/experiment/voice-clone/vits/syned/syn_wav", help="path to target audios"
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
