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
import random
import pdb
import soundfile
warnings.filterwarnings("ignore")

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

############################################################### set the attacks to testing ###############################################################
test_atts = [4,5,6,7,8,9,10,11,12,13,14,15,20,21,22,23,24,25,26]  

start = 4
end = max(test_atts) + 1
return_crop = [4, 5, 6]
return_mel = [21, 23, 25]
dont_need = [16, 17, 18, 19]

# attack_functions = {
#             0: lambda x: self.none(x),
#             1: lambda x: self.crop(x),
#             2: lambda x: self.crop2(x),
#             3: lambda x: self.resample(x),
#             4: lambda x: self.crop_front(x, ratio),     # Cropping front
#             5: lambda x: self.crop_middle(x, ratio),    # Cropping middle
#             6: lambda x: self.crop_back(x, ratio),      # Cropping behind
#             7: lambda x: self.resample1(x),             # Resampling 16KHz
#             8: lambda x: self.resample2(x),             # Resampling 8KHz
#             9: lambda x: self.white_noise(x, ratio),    # Gaussian Noise with SNR ratio/2 dB
#             10: lambda x: self.change_top(x, ratio),    # Amplitude Scaling ratio%
#             11: lambda x: self.mp3(x, ratio),           # MP3 Compression ratio Kbps
#             12: lambda x: self.recount(x),              # Recount 8 bps
#             13: lambda x: self.medfilt(x, ratio),       # Median Filtering with ratio samples as window
#             14: lambda x: self.low_band_pass(x),        # Low Pass Filtering 2000 Hz
#             15: lambda x: self.high_band_pass(x),       # High Pass Filtering 500 Hz 
#             16: lambda x: self.modify_mel(x, ratio),    # don't need
#             17: lambda x: self.crop_mel_front(x, ratio),# don't need        
#             18: lambda x: self.crop_mel_back(x, ratio), # don't need        
#             19: lambda x: self.crop_mel_wave_front(x, ratio),   # don't need
#             20: lambda x: self.crop_mel_wave_back(x, ratio),    # mask from top with ratio "ratio" and transform back to wav
#             21: lambda x: self.crop_mel_position(x, ratio),     # mask 10% at position "ratio"
#             22: lambda x: self.crop_mel_wave_position(x, ratio),# mask 10% at position "ratio" and transform back to wav
            
#             23: lambda x: self.crop_mel_position_5(x, ratio),       # mask 5% at position "ratio"
#             24: lambda x: self.crop_mel_wave_position_5(x, ratio),  # mask 5% at position "ratio" and transform back to wav
#             25: lambda x: self.crop_mel_position_20(x, ratio),      # mask 20% at position "ratio"
#             26: lambda x: self.crop_mel_wave_position_20(x, ratio), # mask 20% at position "ratio" and transform back to wav
#         }


def get_ratio_list(att):
    if att < 4 or att in dont_need:
        return 0
    elif att in [4, 5, 6]: # Cropping front, middle, behind
        ratio_list = [20,40,60,80,90,95,96,97,98]
    elif att in [10]: # Amplitude Scaling
        ratio_list = [20,40,60,80]
    elif att in [9]: # GN
        ratio_list = [40,50,60,70,80]  # snr = 1/2 * ratio
    elif att in [11]: # mp3
        ratio_list = [8,16,24,32,40,48,56,64]
    elif att in [13]: # Median Filtering
        ratio_list = [5, 15, 25, 35]
    elif att in [7,8,12,14,15]: # Resampling, Recount, Filtering
        ratio_list = [0]
    elif att in [20]: # crop-mel-ratio-wave
        ratio_list = [10,20,30,40,50,60,70,80]
    elif att in [21, 22]: # crop-mel-position crop-mel-wave-position
        ratio_list = [1,2,3,4,5,6,7,8,9,10]
    elif att in [23, 24]: # crop-mel-position_5% crop-mel-wave-position_5%
        ratio_list = [ri for ri in range(1,21)]
    elif att in [25, 26]: # crop-mel-position_20% crop-mel-wave-position_20%
        ratio_list = [ri for ri in range(1,6)]
    else:
        raise("Not implementation error")
    return ratio_list



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
        logging.info("use conv2 model")
        from model.conv2_modules import Encoder, Decoder, Discriminator
        from dataset.data import mel_dataset as my_dataset
    elif model_config["structure"]["conv2mel"]:
        if not model_config["structure"]["ab"]:
            logging.info("use conv2mel model")
            from model.conv2_mel_modules import Encoder, Decoder, Discriminator
            from dataset.data import wav_dataset_test as my_dataset
        else:
            logging.info("use ablation conv2mel model")
            from model.conv2_mel_modules_ab import Encoder, Decoder, Discriminator
            from dataset.data import wav_dataset_test as my_dataset
    else:
        from model.conv_modules import Encoder, Decoder
        from dataset.data import oned_dataset as my_dataset
    # ---------------- get train dataset
    audios = my_dataset(process_config=process_config, train_config=train_config,flag='test')
    batch_size = train_config["optimize"]["batch_size"]
    assert batch_size < len(audios)
    # audios_loader = DataLoader(audios, batch_size=batch_size, shuffle=True, collate_fn=audios.collate_fn)
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
        # model_list.remove('train.py')
        model_list = sorted(model_list,key=lambda x:os.path.getmtime(os.path.join(path_model,x)))
        model_path = os.path.join(path_model, model_list[index])
        model = torch.load(model_path)
        logging.info("model <<{}>> loadded".format(model_path))
    # encoder = model["encoder"]
    # decoder = model["decoder"]
    encoder.load_state_dict(model["encoder"])
    decoder.load_state_dict(model["decoder"], strict=False)
    encoder.eval()
    decoder.eval()
    # ---------------- Loss
    loss = Loss(train_config=train_config)

    # ---------------- Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)


    # ---------------- train
    logging.info(logging_mark + "\t" + "Begin Testing" + "\t" + logging_mark)
    epoch_num = train_config["iter"]["epoch"]
    save_circle = train_config["iter"]["save_circle"]
    show_circle = train_config["iter"]["show_circle"]
    lambda_e = train_config["optimize"]["lambda_e"]
    lambda_m = train_config["optimize"]["lambda_m"]
    train_len = len(audios_loader)
    
    from distortions.dl import distortion
    dl = distortion(process_config)
    experiments_dir = args.name



    # initialize results
    all_results = []
    for att in range(0, end):
        all_results.append({})
    for att in test_atts:
        if att < 4 or att in dont_need:
            continue
        ratio_list = get_ratio_list(att)
        now_dir = os.path.join(experiments_dir, str(att))
        if not os.path.exists(now_dir): os.makedirs(now_dir)
        for ratio in ratio_list:
            avg_encode_snr, avg_decode_acc, avg_att_snr, avg_att_pesq, avg_att_utterance_sim, avg_att_spk_sim = 0, 0, 0, 0, 0, 0
            rf = open(os.path.join(now_dir, str(att)+"_"+str(ratio) + ".txt"), 'w')
            all_results[att][str(ratio)] = {
                                            'f':rf, 
                                            'avgs':[avg_decode_acc,
                                                    avg_encode_snr, 
                                                    avg_att_snr, 
                                                    avg_att_pesq, 
                                                    avg_att_utterance_sim, 
                                                    avg_att_spk_sim], 
                                            'example':os.path.join(now_dir, str(att)+"_"+str(ratio)+".wav")
                                            }
    
    # cal fedlity
    from utils.tools import fidelity
    rf = open(os.path.join(experiments_dir, "fidelity.txt"), 'w')
    wmf = open(os.path.join(experiments_dir, "wm.txt"), 'w')
    avg_snr, avg_pesq, avg_utterance_sim, avg_spk_sim = 0, 0, 0, 0
    global_step = 0
    with torch.no_grad():
        for sample in track(audios_loader):
            global_step += 1
            # ---------------- build watermark
            msg = np.random.choice([0,1], [batch_size, 1, msg_length])
            wmf.write(f"{msg.tolist()}\n")
            msg = torch.from_numpy(msg).float()*2 - 1
            wav_matrix = sample["matrix"].to(device)
            sample_rate = sample["sample_rate"]
            name = sample["name"]
            msg = msg.to(device)
            
            encoded, carrier_wateramrked = encoder.test_forward(wav_matrix, msg)
            decoded = decoder.test_forward(encoded)
            losses = loss.en_de_loss(wav_matrix, encoded, msg, decoded)
            decoder_acc = (decoded >= 0).eq(msg >= 0).sum().float() / msg.numel()
            snr_score, pesq_score, utterance_sim_matrix, spk_sim_matrix = fidelity(wav_matrix.detach(), encoded.detach(), process_config["audio"]["sample_rate"])
            zero_tensor = torch.zeros(wav_matrix.shape).to(device)
            norm2=mse_loss(wav_matrix.detach(),zero_tensor)
            avg_pesq += pesq_score
            avg_snr += snr_score
            avg_spk_sim += spk_sim_matrix
            avg_utterance_sim += utterance_sim_matrix
            rf.write("audio:{},\tsnr:{:.8f}, pesq:{:.8f}, utterance_sim:{:.8f}, spk_sim:{:.8f}\n".format(name, snr_score, pesq_score, utterance_sim_matrix, spk_sim_matrix))
            logging.info('-' * 100)
            logging.info("step:{} - wav_loss:{:.8f} - msg_loss:{:.8f} - acc:{:.8f} - snr:{:.8f} - pesq:{:.8f} - utterance_sim:{:.8f}- spk_sim:{:.8f}  - wav_len:{} - name:{}".format( \
                global_step, losses[0], losses[1], decoder_acc, snr_score, pesq_score, utterance_sim_matrix, spk_sim_matrix, wav_matrix.shape[2], sample["name"][0]))
            
            # test distortions
            for att in test_atts:
                if att < 4 or att in dont_need:
                    continue
                ratio_list = get_ratio_list(att)
                for ratio in ratio_list:
                    distored = dl(encoded, att, ratio)
                    if att in return_mel:
                        decoded = decoder.mel_test_forward(distored)
                    else:
                        decoded = decoder.test_forward(distored)
                    decoder_acc = (decoded >= 0).eq(msg >= 0).sum().float() / msg.numel()
                    all_results[att][str(ratio)]['avgs'][0] += decoder_acc.item()
                    all_results[att][str(ratio)]['avgs'][1] += snr_score
                    if att not in return_crop+return_mel:
                        att_snr, att_pesq, att_utterance_sim, att_spk_sim = fidelity(encoded.detach(), distored.detach(), process_config["audio"]["sample_rate"])
                    else:
                        att_snr, att_pesq, att_utterance_sim, att_spk_sim = 0, 0, 0, 0
                    all_results[att][str(ratio)]['avgs'][2] += att_snr
                    all_results[att][str(ratio)]['avgs'][3] += att_pesq
                    all_results[att][str(ratio)]['avgs'][4] += att_utterance_sim
                    all_results[att][str(ratio)]['avgs'][5] += att_spk_sim
                    all_results[att][str(ratio)]['f'].write("audio:{},\tsnr:{:.8f}, acc:{:.8f}, att_snr:{:.8f}, att_pesq:{:.8f}, att_utterance_sim:{:.8f}, att_spk_sim:{:.8f}\n".format(\
                                                                    name, snr_score.item(), decoder_acc, att_snr, att_pesq, att_utterance_sim, att_spk_sim))
                    if global_step == 1 and att not in return_mel: # not wav form
                        soundfile.write(all_results[att][str(ratio)]['example'], distored.cpu().squeeze(0).squeeze(0).detach().numpy(), samplerate=process_config["audio"]["sample_rate"])
                    logging.info('-' * 100)
                    logging.info("step:{} - acc:{:.8f} - snr:{:.8f} - att_snr:{:.8f} - att_pesq:{:.8f} - att_utterance_sim:{:.8f} - att_spk_sim:{:.8f} - name:{}".format( \
                        global_step, decoder_acc, snr_score, att_snr, att_pesq, att_utterance_sim, att_spk_sim, sample["name"][0]))
                    
        rf.write("{} audios, avg_snr:{:.8f}, avg_pesq:{:.8f}, utterance_sim:{:.8f}, spk_sim:{:.8f}".format(global_step, avg_snr/global_step, avg_pesq/global_step, avg_utterance_sim/global_step, avg_spk_sim/global_step))
        rf.close()
        wmf.close()
    
    # close txt
    for att in test_atts:
        if att < 4 or att in dont_need:
                    continue
        ratio_list = get_ratio_list(att)
        for ratio in ratio_list:
            temp = np.array(all_results[att][str(ratio)]['avgs'])/global_step
            all_results[att][str(ratio)]['f'].write("{} audios, avg_acc:{:.8f}, avg_snr:{:.8f}, avg_att_snr:{:.8f}, avg_pesq:{:.8f}, utterance_sim:{:.8f}, spk_sim:{:.8f}\n".format(global_step, temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]))
            all_results[att][str(ratio)]['f'].write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t".format(temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]))
            all_results[att][str(ratio)]['f'].close()
    



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
        "-n", "--name", type=str, default="experiments_results", help="path to save results"
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