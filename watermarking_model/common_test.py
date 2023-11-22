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
    # experiments_dir = 'experiments_results'
    experiments_dir = args.name
    start = 4
    end = 23
    all_results = []
    # for att in range(0, end):
    #     all_results.append({})
    for att in range(0, end):
        all_results.append({})
        if att in [4,5,6]:
            now_dir = os.path.join(experiments_dir, str(att))
            if not os.path.exists(now_dir): os.makedirs(now_dir)
            for ratio in [20,40,60,80,90,95,96,97,98]:
                # all_results[att].append({})
                avg_encode_snr, avg_decode_acc, avg_att_snr, avg_att_pesq, avg_att_utterance_sim, avg_att_spk_sim = 0, 0, 0, 0, 0, 0
                rf = open(os.path.join(now_dir, str(att)+"_"+str(ratio) + ".txt"), 'w')
                all_results[att][str(ratio)] = {'f':rf, 'avgs':[avg_encode_snr, avg_decode_acc, avg_att_snr, avg_att_pesq, avg_att_utterance_sim, avg_att_spk_sim], 'example':os.path.join(now_dir, str(att)+"_"+str(ratio)+".wav")}
        elif att in [10]:
            now_dir = os.path.join(experiments_dir, str(att))
            if not os.path.exists(now_dir): os.makedirs(now_dir)
            for ratio in [20,40,60,80]:
                # all_results[att].append({})
                avg_encode_snr, avg_decode_acc, avg_att_snr, avg_att_pesq, avg_att_utterance_sim, avg_att_spk_sim = 0, 0, 0, 0, 0, 0
                rf = open(os.path.join(now_dir, str(att)+"_"+str(ratio) + ".txt"), 'w')
                all_results[att][str(ratio)] = {'f':rf, 'avgs':[avg_encode_snr, avg_decode_acc, avg_att_snr, avg_att_pesq, avg_att_utterance_sim, avg_att_spk_sim], 'example':os.path.join(now_dir, str(att)+"_"+str(ratio)+".wav")}
        elif att in [9]: # gn
            now_dir = os.path.join(experiments_dir, str(att))
            if not os.path.exists(now_dir): os.makedirs(now_dir)
            for ratio in [10,20,30,40,50,60,70,80]:
                # all_results[att].append({})
                avg_encode_snr, avg_decode_acc, avg_att_snr, avg_att_pesq, avg_att_utterance_sim, avg_att_spk_sim = 0, 0, 0, 0, 0, 0
                rf = open(os.path.join(now_dir, str(att)+"_"+str(ratio) + ".txt"), 'w')
                all_results[att][str(ratio)] = {'f':rf, 'avgs':[avg_encode_snr, avg_decode_acc, avg_att_snr, avg_att_pesq, avg_att_utterance_sim, avg_att_spk_sim], 'example':os.path.join(now_dir, str(att)+"_"+str(ratio)+".wav")}
        elif att in [11]: #mp3
            now_dir = os.path.join(experiments_dir, str(att))
            if not os.path.exists(now_dir): os.makedirs(now_dir)
            for ratio in [8,16,24,32,40,48,56,64]:
                # all_results[att].append({})
                avg_encode_snr, avg_decode_acc, avg_att_snr, avg_att_pesq, avg_att_utterance_sim, avg_att_spk_sim = 0, 0, 0, 0, 0, 0
                rf = open(os.path.join(now_dir, str(att)+"_"+str(ratio) + ".txt"), 'w')
                all_results[att][str(ratio)] = {'f':rf, 'avgs':[avg_encode_snr, avg_decode_acc, avg_att_snr, avg_att_pesq, avg_att_utterance_sim, avg_att_spk_sim], 'example':os.path.join(now_dir, str(att)+"_"+str(ratio)+".wav")}
        elif att in [13]: #medianfilt
            now_dir = os.path.join(experiments_dir, str(att))
            if not os.path.exists(now_dir): os.makedirs(now_dir)
            for ratio in [5, 15, 25, 35]:
                # all_results[att].append({})
                avg_encode_snr, avg_decode_acc, avg_att_snr, avg_att_pesq, avg_att_utterance_sim, avg_att_spk_sim = 0, 0, 0, 0, 0, 0
                rf = open(os.path.join(now_dir, str(att)+"_"+str(ratio) + ".txt"), 'w')
                all_results[att][str(ratio)] = {'f':rf, 'avgs':[avg_encode_snr, avg_decode_acc, avg_att_snr, avg_att_pesq, avg_att_utterance_sim, avg_att_spk_sim], 'example':os.path.join(now_dir, str(att)+"_"+str(ratio)+".wav")}
        elif att in [7,8,12,14,15]:
            now_dir = os.path.join(experiments_dir, str(att))
            if not os.path.exists(now_dir): os.makedirs(now_dir)
            ratio = 0
            # all_results[att].append({})
            avg_encode_snr, avg_decode_acc, avg_att_snr, avg_att_pesq, avg_att_utterance_sim, avg_att_spk_sim = 0, 0, 0, 0, 0, 0
            rf = open(os.path.join(now_dir, str(att)+"_"+str(ratio) + ".txt"), 'w')
            all_results[att][str(ratio)] = {'f':rf, 'avgs':[avg_encode_snr, avg_decode_acc, avg_att_snr, avg_att_pesq, avg_att_utterance_sim, avg_att_spk_sim], 'example':os.path.join(now_dir, str(att)+"_"+str(ratio)+".wav")}
        # elif att in [17, 18]: # crop-mel
        #     for ratio in [10,20,30,40,50,60,70,80]:
        #         # all_results[att].append({})
        #         avg_encode_snr, avg_decode_acc, avg_att_snr, avg_att_pesq, avg_att_utterance_sim, avg_att_spk_sim = 0, 0, 0, 0, 0, 0
        #         rf = open(os.path.join(now_dir, str(att)+"_"+str(ratio) + ".txt"), 'w')
        #         all_results[att][str(ratio)] = {'f':rf, 'avgs':[avg_encode_snr, avg_decode_acc, avg_att_snr, avg_att_pesq, avg_att_utterance_sim, avg_att_spk_sim], 'example':os.path.join(now_dir, str(att)+"_"+str(ratio)+".wav")}
        # elif att in [19,20]: # crop-mel-wave
        #     for ratio in [10,20,30,40,50,60,70,80]:
        #         # all_results[att].append({})
        #         avg_encode_snr, avg_decode_acc, avg_att_snr, avg_att_pesq, avg_att_utterance_sim, avg_att_spk_sim = 0, 0, 0, 0, 0, 0
        #         rf = open(os.path.join(now_dir, str(att)+"_"+str(ratio) + ".txt"), 'w')
        #         all_results[att][str(ratio)] = {'f':rf, 'avgs':[avg_encode_snr, avg_decode_acc, avg_att_snr, avg_att_pesq, avg_att_utterance_sim, avg_att_spk_sim], 'example':os.path.join(now_dir, str(att)+"_"+str(ratio)+".wav")}
        elif att < 4 or att in [16, 17, 18, 19, 20]:
            pass
        elif att in [21]: # crop-mel-position
            now_dir = os.path.join(experiments_dir, str(att))
            if not os.path.exists(now_dir): os.makedirs(now_dir)
            for ratio in [1,2,3,4,5,6,7,8,9,10]:
                # all_results[att].append({})
                avg_encode_snr, avg_decode_acc, avg_att_snr, avg_att_pesq, avg_att_utterance_sim, avg_att_spk_sim = 0, 0, 0, 0, 0, 0
                rf = open(os.path.join(now_dir, str(att)+"_"+str(ratio) + ".txt"), 'w')
                all_results[att][str(ratio)] = {'f':rf, 'avgs':[avg_encode_snr, avg_decode_acc, avg_att_snr, avg_att_pesq, avg_att_utterance_sim, avg_att_spk_sim], 'example':os.path.join(now_dir, str(att)+"_"+str(ratio)+".wav")}
        elif att in [22]: # crop-mel-wave-position
            now_dir = os.path.join(experiments_dir, str(att))
            if not os.path.exists(now_dir): os.makedirs(now_dir)
            for ratio in [1,2,3,4,5,6,7,8,9,10]:
                # all_results[att].append({})
                avg_encode_snr, avg_decode_acc, avg_att_snr, avg_att_pesq, avg_att_utterance_sim, avg_att_spk_sim = 0, 0, 0, 0, 0, 0
                rf = open(os.path.join(now_dir, str(att)+"_"+str(ratio) + ".txt"), 'w')
                all_results[att][str(ratio)] = {'f':rf, 'avgs':[avg_encode_snr, avg_decode_acc, avg_att_snr, avg_att_pesq, avg_att_utterance_sim, avg_att_spk_sim], 'example':os.path.join(now_dir, str(att)+"_"+str(ratio)+".wav")}
        elif att in [23]: # crop-mel-position
            now_dir = os.path.join(experiments_dir, str(att))
            if not os.path.exists(now_dir): os.makedirs(now_dir)
            for ratio in range(1, 21):
                # all_results[att].append({})
                avg_encode_snr, avg_decode_acc, avg_att_snr, avg_att_pesq, avg_att_utterance_sim, avg_att_spk_sim = 0, 0, 0, 0, 0, 0
                rf = open(os.path.join(now_dir, str(att)+"_"+str(ratio) + ".txt"), 'w')
                all_results[att][str(ratio)] = {'f':rf, 'avgs':[avg_encode_snr, avg_decode_acc, avg_att_snr, avg_att_pesq, avg_att_utterance_sim, avg_att_spk_sim], 'example':os.path.join(now_dir, str(att)+"_"+str(ratio)+".wav")}
        elif att in [24]: # crop-mel-wave-position
            now_dir = os.path.join(experiments_dir, str(att))
            if not os.path.exists(now_dir): os.makedirs(now_dir)
            for ratio in range(1, 21):
                # all_results[att].append({})
                avg_encode_snr, avg_decode_acc, avg_att_snr, avg_att_pesq, avg_att_utterance_sim, avg_att_spk_sim = 0, 0, 0, 0, 0, 0
                rf = open(os.path.join(now_dir, str(att)+"_"+str(ratio) + ".txt"), 'w')
                all_results[att][str(ratio)] = {'f':rf, 'avgs':[avg_encode_snr, avg_decode_acc, avg_att_snr, avg_att_pesq, avg_att_utterance_sim, avg_att_spk_sim], 'example':os.path.join(now_dir, str(att)+"_"+str(ratio)+".wav")}
        elif att in [25]: # crop-mel-position
            now_dir = os.path.join(experiments_dir, str(att))
            if not os.path.exists(now_dir): os.makedirs(now_dir)
            for ratio in range(1, 6):
                # all_results[att].append({})
                avg_encode_snr, avg_decode_acc, avg_att_snr, avg_att_pesq, avg_att_utterance_sim, avg_att_spk_sim = 0, 0, 0, 0, 0, 0
                rf = open(os.path.join(now_dir, str(att)+"_"+str(ratio) + ".txt"), 'w')
                all_results[att][str(ratio)] = {'f':rf, 'avgs':[avg_encode_snr, avg_decode_acc, avg_att_snr, avg_att_pesq, avg_att_utterance_sim, avg_att_spk_sim], 'example':os.path.join(now_dir, str(att)+"_"+str(ratio)+".wav")}
        elif att in [26]: # crop-mel-wave-position
            now_dir = os.path.join(experiments_dir, str(att))
            if not os.path.exists(now_dir): os.makedirs(now_dir)
            for ratio in range(1, 6):
                # all_results[att].append({})
                avg_encode_snr, avg_decode_acc, avg_att_snr, avg_att_pesq, avg_att_utterance_sim, avg_att_spk_sim = 0, 0, 0, 0, 0, 0
                rf = open(os.path.join(now_dir, str(att)+"_"+str(ratio) + ".txt"), 'w')
                all_results[att][str(ratio)] = {'f':rf, 'avgs':[avg_encode_snr, avg_decode_acc, avg_att_snr, avg_att_pesq, avg_att_utterance_sim, avg_att_spk_sim], 'example':os.path.join(now_dir, str(att)+"_"+str(ratio)+".wav")}
        else:
            raise("Not implementation error")
    
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
            # snr = 10 * torch.log10(mse_loss(wav_matrix.detach(), zero_tensor) / mse_loss(wav_matrix.detach(), encoded.detach()))
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
            
            for att in range(start, end):
                if att in [4,5,6]:
                    for ratio in [20,40,60,80,90,95,96,97,98]:
                        distored = dl(encoded, att, ratio)
                        decoded = decoder.test_forward(distored)
                        decoder_acc = (decoded >= 0).eq(msg >= 0).sum().float() / msg.numel()
                        all_results[att][str(ratio)]['avgs'][0] += decoder_acc.item()
                        all_results[att][str(ratio)]['avgs'][1] += snr_score

                        if att not in [4,5,6]:
                            att_snr, att_pesq, att_utterance_sim, att_spk_sim = fidelity(encoded.detach(), distored.detach(), process_config["audio"]["sample_rate"])
                        else:
                            att_snr, att_pesq, att_utterance_sim, att_spk_sim = 0,0,0,0
                        all_results[att][str(ratio)]['avgs'][2] += att_snr
                        all_results[att][str(ratio)]['avgs'][3] += att_pesq
                        all_results[att][str(ratio)]['avgs'][4] += att_utterance_sim
                        all_results[att][str(ratio)]['avgs'][5] += att_spk_sim
                        all_results[att][str(ratio)]['f'].write("audio:{},\tsnr:{:.8f}, acc:{:.8f}, att_snr:{:.8f}, att_pesq:{:.8f}, att_utterance_sim:{:.8f}, att_spk_sim:{:.8f}\n".format(name, snr_score.item(), decoder_acc, att_snr, att_pesq, att_utterance_sim, att_spk_sim))
                        if global_step == 1:
                            soundfile.write(all_results[att][str(ratio)]['example'], distored.cpu().squeeze(0).squeeze(0).detach().numpy(), samplerate=process_config["audio"]["sample_rate"])
                        logging.info('-' * 100)
                        logging.info("step:{} - acc:{:.8f} - snr:{:.8f} - att_snr:{:.8f} - att_pesq:{:.8f} - att_utterance_sim:{:.8f} - att_spk_sim:{:.8f} - name:{}".format( \
                            global_step, decoder_acc, snr_score, att_snr, att_pesq, att_utterance_sim, att_spk_sim, sample["name"][0]))
                elif att in [10]:
                    for ratio in [20,40,60,80]:
                        distored = dl(encoded, att, ratio)
                        decoded = decoder.test_forward(distored)
                        decoder_acc = (decoded >= 0).eq(msg >= 0).sum().float() / msg.numel()
                        all_results[att][str(ratio)]['avgs'][0] += decoder_acc.item()
                        all_results[att][str(ratio)]['avgs'][1] += snr_score

                        if att not in [4,5,6]:
                            att_snr, att_pesq, att_utterance_sim, att_spk_sim = fidelity(encoded.detach(), distored.detach(), process_config["audio"]["sample_rate"])
                        else:
                            att_snr, att_pesq, att_utterance_sim, att_spk_sim = 0,0,0,0
                        all_results[att][str(ratio)]['avgs'][2] += att_snr
                        all_results[att][str(ratio)]['avgs'][3] += att_pesq
                        all_results[att][str(ratio)]['avgs'][4] += att_utterance_sim
                        all_results[att][str(ratio)]['avgs'][5] += att_spk_sim
                        all_results[att][str(ratio)]['f'].write("audio:{},\tsnr:{:.8f}, acc:{:.8f}, att_snr:{:.8f}, att_pesq:{:.8f}, att_utterance_sim:{:.8f}, att_spk_sim:{:.8f}\n".format(name, snr_score.item(), decoder_acc, att_snr, att_pesq, att_utterance_sim, att_spk_sim))
                        if global_step == 1:
                            soundfile.write(all_results[att][str(ratio)]['example'], distored.cpu().squeeze(0).squeeze(0).detach().numpy(), samplerate=process_config["audio"]["sample_rate"])
                        logging.info('-' * 100)
                        logging.info("step:{} - acc:{:.8f} - snr:{:.8f} - att_snr:{:.8f} - att_pesq:{:.8f} - att_utterance_sim:{:.8f} - att_spk_sim:{:.8f} - name:{}".format( \
                            global_step, decoder_acc, snr_score, att_snr, att_pesq, att_utterance_sim, att_spk_sim, sample["name"][0]))
                elif att in [9]: # gn
                    for ratio in [40,50,60,70,80]: # snr = 1/2 * ratio
                        distored = dl(encoded, att, ratio)
                        decoded = decoder.test_forward(distored)
                        decoder_acc = (decoded >= 0).eq(msg >= 0).sum().float() / msg.numel()
                        all_results[att][str(ratio)]['avgs'][0] += decoder_acc.item()
                        all_results[att][str(ratio)]['avgs'][1] += snr_score

                        att_snr, att_pesq, att_utterance_sim, att_spk_sim = fidelity(encoded.detach(), distored.detach(), process_config["audio"]["sample_rate"])
                        all_results[att][str(ratio)]['avgs'][2] += att_snr
                        all_results[att][str(ratio)]['avgs'][3] += att_pesq
                        all_results[att][str(ratio)]['avgs'][4] += att_utterance_sim
                        all_results[att][str(ratio)]['avgs'][5] += att_spk_sim
                        all_results[att][str(ratio)]['f'].write("audio:{},\tsnr:{:.8f}, acc:{:.8f}, att_snr:{:.8f}, att_pesq:{:.8f}, att_utterance_sim:{:.8f}, att_spk_sim:{:.8f}\n".format(name, snr_score.item(), decoder_acc, att_snr, att_pesq, att_utterance_sim, att_spk_sim))
                        if global_step == 1:
                            soundfile.write(all_results[att][str(ratio)]['example'], distored.cpu().squeeze(0).squeeze(0).detach().numpy(), samplerate=process_config["audio"]["sample_rate"])
                        logging.info('-' * 100)
                        logging.info("step:{} - acc:{:.8f} - snr:{:.8f} - att_snr:{:.8f} - att_pesq:{:.8f} - att_utterance_sim:{:.8f} - att_spk_sim:{:.8f} - name:{}".format( \
                            global_step, decoder_acc, snr_score, att_snr, att_pesq, att_utterance_sim, att_spk_sim, sample["name"][0]))
                elif att in [11]: #mp3
                    for ratio in [8,16,24,32,40,48,56,64]:
                        distored = dl(encoded, att, ratio)
                        decoded = decoder.test_forward(distored)
                        decoder_acc = (decoded >= 0).eq(msg >= 0).sum().float() / msg.numel()
                        all_results[att][str(ratio)]['avgs'][0] += decoder_acc.item()
                        all_results[att][str(ratio)]['avgs'][1] += snr_score

                        att_snr, att_pesq, att_utterance_sim, att_spk_sim = fidelity(encoded.detach(), distored.detach(), process_config["audio"]["sample_rate"])
                        all_results[att][str(ratio)]['avgs'][2] += att_snr
                        all_results[att][str(ratio)]['avgs'][3] += att_pesq
                        all_results[att][str(ratio)]['avgs'][4] += att_utterance_sim
                        all_results[att][str(ratio)]['avgs'][5] += att_spk_sim
                        all_results[att][str(ratio)]['f'].write("audio:{},\tsnr:{:.8f}, acc:{:.8f}, att_snr:{:.8f}, att_pesq:{:.8f}, att_utterance_sim:{:.8f}, att_spk_sim:{:.8f}\n".format(name, snr_score.item(), decoder_acc, att_snr, att_pesq, att_utterance_sim, att_spk_sim))
                        if global_step == 1:
                            soundfile.write(all_results[att][str(ratio)]['example'], distored.cpu().squeeze(0).squeeze(0).detach().numpy(), samplerate=process_config["audio"]["sample_rate"])
                        logging.info('-' * 100)
                        logging.info("step:{} - acc:{:.8f} - snr:{:.8f} - att_snr:{:.8f} - att_pesq:{:.8f} - att_utterance_sim:{:.8f} - att_spk_sim:{:.8f} - name:{}".format( \
                            global_step, decoder_acc, snr_score, att_snr, att_pesq, att_utterance_sim, att_spk_sim, sample["name"][0]))
                elif att in [13]: #medianfilt
                    for ratio in [5, 15, 25, 35]:
                        distored = dl(encoded, att, ratio)
                        decoded = decoder.test_forward(distored)
                        decoder_acc = (decoded >= 0).eq(msg >= 0).sum().float() / msg.numel()
                        all_results[att][str(ratio)]['avgs'][0] += decoder_acc.item()
                        all_results[att][str(ratio)]['avgs'][1] += snr_score

                        att_snr, att_pesq, att_utterance_sim, att_spk_sim = fidelity(encoded.detach(), distored.detach(), process_config["audio"]["sample_rate"])
                        all_results[att][str(ratio)]['avgs'][2] += att_snr
                        all_results[att][str(ratio)]['avgs'][3] += att_pesq
                        all_results[att][str(ratio)]['avgs'][4] += att_utterance_sim
                        all_results[att][str(ratio)]['avgs'][5] += att_spk_sim
                        all_results[att][str(ratio)]['f'].write("audio:{},\tsnr:{:.8f}, acc:{:.8f}, att_snr:{:.8f}, att_pesq:{:.8f}, att_utterance_sim:{:.8f}, att_spk_sim:{:.8f}\n".format(name, snr_score.item(), decoder_acc, att_snr, att_pesq, att_utterance_sim, att_spk_sim))
                        if global_step == 1:
                            soundfile.write(all_results[att][str(ratio)]['example'], distored.cpu().squeeze(0).squeeze(0).detach().numpy(), samplerate=process_config["audio"]["sample_rate"])
                        logging.info('-' * 100)
                        logging.info("step:{} - acc:{:.8f} - snr:{:.8f} - att_snr:{:.8f} - att_pesq:{:.8f} - att_utterance_sim:{:.8f} - att_spk_sim:{:.8f} - name:{}".format( \
                            global_step, decoder_acc, snr_score, att_snr, att_pesq, att_utterance_sim, att_spk_sim, sample["name"][0]))
                elif att in [7,8,12,14,15]:
                    ratio = 0
                    distored = dl(encoded, att, ratio)
                    decoded = decoder.test_forward(distored)
                    decoder_acc = (decoded >= 0).eq(msg >= 0).sum().float() / msg.numel()
                    all_results[att][str(ratio)]['avgs'][0] += decoder_acc.item()
                    all_results[att][str(ratio)]['avgs'][1] += snr_score

                    att_snr, att_pesq, att_utterance_sim, att_spk_sim = fidelity(encoded.detach(), distored.detach(), process_config["audio"]["sample_rate"])
                    all_results[att][str(ratio)]['avgs'][2] += att_snr
                    all_results[att][str(ratio)]['avgs'][3] += att_pesq
                    all_results[att][str(ratio)]['avgs'][4] += att_utterance_sim
                    all_results[att][str(ratio)]['avgs'][5] += att_spk_sim
                    all_results[att][str(ratio)]['f'].write("audio:{},\tsnr:{:.8f}, acc:{:.8f}, att_snr:{:.8f}, att_pesq:{:.8f}, att_utterance_sim:{:.8f}, att_spk_sim:{:.8f}\n".format(name, snr_score.item(), decoder_acc, att_snr, att_pesq, att_utterance_sim, att_spk_sim))
                    if global_step == 1:
                        soundfile.write(all_results[att][str(ratio)]['example'], distored.cpu().squeeze(0).squeeze(0).detach().numpy(), samplerate=process_config["audio"]["sample_rate"])
                    logging.info('-' * 100)
                    logging.info("step:{} - acc:{:.8f} - snr:{:.8f} - att_snr:{:.8f} - att_pesq:{:.8f} - att_utterance_sim:{:.8f} - att_spk_sim:{:.8f} - name:{}".format( \
                        global_step, decoder_acc, snr_score, att_snr, att_pesq, att_utterance_sim, att_spk_sim, sample["name"][0]))
                # elif att in [17, 18]: # crop-mel
                #     for ratio in [10,20,30,40,50,60,70,80,90]:
                #         distored = dl(encoded, att, ratio)
                #         decoded = decoder.mel_test_forward(distored)
                #         decoder_acc = (decoded >= 0).eq(msg >= 0).sum().float() / msg.numel()
                #         all_results[att][str(ratio)]['avgs'][0] += decoder_acc.item()
                #         all_results[att][str(ratio)]['avgs'][1] += snr_score

                #         att_snr, att_pesq, att_utterance_sim, att_spk_sim = 0,0,0,0
                #         all_results[att][str(ratio)]['avgs'][2] += att_snr
                #         all_results[att][str(ratio)]['avgs'][3] += att_pesq
                #         all_results[att][str(ratio)]['avgs'][4] += att_utterance_sim
                #         all_results[att][str(ratio)]['avgs'][5] += att_spk_sim
                #         all_results[att][str(ratio)]['f'].write("audio:{},\tsnr:{:.8f}, acc:{:.8f}, att_snr:{:.8f}, att_pesq:{:.8f}, att_utterance_sim:{:.8f}, att_spk_sim:{:.8f}\n".format(name, snr_score.item(), decoder_acc, att_snr, att_pesq, att_utterance_sim, att_spk_sim))
                #         # if global_step == 1:
                #         #     soundfile.write(all_results[att][str(ratio)]['example'], distored.cpu().squeeze(0).squeeze(0).detach().numpy(), samplerate=process_config["audio"]["sample_rate"])
                #         logging.info('-' * 100)
                #         logging.info("step:{} - acc:{:.8f} - snr:{:.8f} - att_snr:{:.8f} - att_pesq:{:.8f} - att_utterance_sim:{:.8f} - att_spk_sim:{:.8f} - name:{}".format( \
                #             global_step, decoder_acc, snr_score, att_snr, att_pesq, att_utterance_sim, att_spk_sim, sample["name"][0]))
                # elif att in [19,20]: # crop-mel-wave
                #     for ratio in [10,20,30,40,50,60,70,80,90]:
                #         distored = dl(encoded, att, ratio)
                #         decoded = decoder.test_forward(distored)
                #         decoder_acc = (decoded >= 0).eq(msg >= 0).sum().float() / msg.numel()
                #         all_results[att][str(ratio)]['avgs'][0] += decoder_acc.item()
                #         all_results[att][str(ratio)]['avgs'][1] += snr_score

                #         att_snr, att_pesq, att_utterance_sim, att_spk_sim = fidelity(encoded.detach(), distored.detach(), process_config["audio"]["sample_rate"])
                #         all_results[att][str(ratio)]['avgs'][2] += att_snr
                #         all_results[att][str(ratio)]['avgs'][3] += att_pesq
                #         all_results[att][str(ratio)]['avgs'][4] += att_utterance_sim
                #         all_results[att][str(ratio)]['avgs'][5] += att_spk_sim
                #         all_results[att][str(ratio)]['f'].write("audio:{},\tsnr:{:.8f}, acc:{:.8f}, att_snr:{:.8f}, att_pesq:{:.8f}, att_utterance_sim:{:.8f}, att_spk_sim:{:.8f}\n".format(name, snr_score.item(), decoder_acc, att_snr, att_pesq, att_utterance_sim, att_spk_sim))
                #         if global_step == 1:
                #             soundfile.write(all_results[att][str(ratio)]['example'], distored.cpu().squeeze(0).squeeze(0).detach().numpy(), samplerate=process_config["audio"]["sample_rate"])
                #         logging.info('-' * 100)
                #         logging.info("step:{} - acc:{:.8f} - snr:{:.8f} - att_snr:{:.8f} - att_pesq:{:.8f} - att_utterance_sim:{:.8f} - att_spk_sim:{:.8f} - name:{}".format( \
                #             global_step, decoder_acc, snr_score, att_snr, att_pesq, att_utterance_sim, att_spk_sim, sample["name"][0]))
                elif att in [21]: # crop-mel-position
                    for ratio in [1,2,3,4,5,6,7,8,9,10]:
                        distored = dl(encoded, att, ratio)
                        decoded = decoder.mel_test_forward(distored)
                        decoder_acc = (decoded >= 0).eq(msg >= 0).sum().float() / msg.numel()
                        all_results[att][str(ratio)]['avgs'][0] += decoder_acc.item()
                        all_results[att][str(ratio)]['avgs'][1] += snr_score

                        att_snr, att_pesq, att_utterance_sim, att_spk_sim = 0,0,0,0
                        all_results[att][str(ratio)]['avgs'][2] += att_snr
                        all_results[att][str(ratio)]['avgs'][3] += att_pesq
                        all_results[att][str(ratio)]['avgs'][4] += att_utterance_sim
                        all_results[att][str(ratio)]['avgs'][5] += att_spk_sim
                        all_results[att][str(ratio)]['f'].write("audio:{},\tsnr:{:.8f}, acc:{:.8f}, att_snr:{:.8f}, att_pesq:{:.8f}, att_utterance_sim:{:.8f}, att_spk_sim:{:.8f}\n".format(name, snr_score.item(), decoder_acc, att_snr, att_pesq, att_utterance_sim, att_spk_sim))
                        # if global_step == 1:
                        #     soundfile.write(all_results[att][str(ratio)]['example'], distored.cpu().squeeze(0).squeeze(0).detach().numpy(), samplerate=process_config["audio"]["sample_rate"])
                        logging.info('-' * 100)
                        logging.info("step:{} - acc:{:.8f} - snr:{:.8f} - att_snr:{:.8f} - att_pesq:{:.8f} - att_utterance_sim:{:.8f} - att_spk_sim:{:.8f} - name:{}".format( \
                            global_step, decoder_acc, snr_score, att_snr, att_pesq, att_utterance_sim, att_spk_sim, sample["name"][0]))
                elif att in [22]: # crop-mel-wave-position
                    for ratio in [1,2,3,4,5,6,7,8,9,10]:
                        distored = dl(encoded, att, ratio)
                        decoded = decoder.test_forward(distored)
                        decoder_acc = (decoded >= 0).eq(msg >= 0).sum().float() / msg.numel()
                        all_results[att][str(ratio)]['avgs'][0] += decoder_acc.item()
                        all_results[att][str(ratio)]['avgs'][1] += snr_score

                        att_snr, att_pesq, att_utterance_sim, att_spk_sim = fidelity(encoded.detach(), distored.detach(), process_config["audio"]["sample_rate"])
                        all_results[att][str(ratio)]['avgs'][2] += att_snr
                        all_results[att][str(ratio)]['avgs'][3] += att_pesq
                        all_results[att][str(ratio)]['avgs'][4] += att_utterance_sim
                        all_results[att][str(ratio)]['avgs'][5] += att_spk_sim
                        all_results[att][str(ratio)]['f'].write("audio:{},\tsnr:{:.8f}, acc:{:.8f}, att_snr:{:.8f}, att_pesq:{:.8f}, att_utterance_sim:{:.8f}, att_spk_sim:{:.8f}\n".format(name, snr_score.item(), decoder_acc, att_snr, att_pesq, att_utterance_sim, att_spk_sim))
                        if global_step == 1:
                            soundfile.write(all_results[att][str(ratio)]['example'], distored.cpu().squeeze(0).squeeze(0).detach().numpy(), samplerate=process_config["audio"]["sample_rate"])
                        logging.info('-' * 100)
                        logging.info("step:{} - acc:{:.8f} - snr:{:.8f} - att_snr:{:.8f} - att_pesq:{:.8f} - att_utterance_sim:{:.8f} - att_spk_sim:{:.8f} - name:{}".format( \
                            global_step, decoder_acc, snr_score, att_snr, att_pesq, att_utterance_sim, att_spk_sim, sample["name"][0]))
                        
                elif att in [23]: # crop-mel-position
                    for ratio in range(1, 21):
                        distored = dl(encoded, att, ratio)
                        decoded = decoder.mel_test_forward(distored)
                        decoder_acc = (decoded >= 0).eq(msg >= 0).sum().float() / msg.numel()
                        all_results[att][str(ratio)]['avgs'][0] += decoder_acc.item()
                        all_results[att][str(ratio)]['avgs'][1] += snr_score

                        att_snr, att_pesq, att_utterance_sim, att_spk_sim = 0,0,0,0
                        all_results[att][str(ratio)]['avgs'][2] += att_snr
                        all_results[att][str(ratio)]['avgs'][3] += att_pesq
                        all_results[att][str(ratio)]['avgs'][4] += att_utterance_sim
                        all_results[att][str(ratio)]['avgs'][5] += att_spk_sim
                        all_results[att][str(ratio)]['f'].write("audio:{},\tsnr:{:.8f}, acc:{:.8f}, att_snr:{:.8f}, att_pesq:{:.8f}, att_utterance_sim:{:.8f}, att_spk_sim:{:.8f}\n".format(name, snr_score.item(), decoder_acc, att_snr, att_pesq, att_utterance_sim, att_spk_sim))
                        # if global_step == 1:
                        #     soundfile.write(all_results[att][str(ratio)]['example'], distored.cpu().squeeze(0).squeeze(0).detach().numpy(), samplerate=process_config["audio"]["sample_rate"])
                        logging.info('-' * 100)
                        logging.info("step:{} - acc:{:.8f} - snr:{:.8f} - att_snr:{:.8f} - att_pesq:{:.8f} - att_utterance_sim:{:.8f} - att_spk_sim:{:.8f} - name:{}".format( \
                            global_step, decoder_acc, snr_score, att_snr, att_pesq, att_utterance_sim, att_spk_sim, sample["name"][0]))
                elif att in [24]: # crop-mel-wave-position
                    for ratio in range(1, 21):
                        distored = dl(encoded, att, ratio)
                        decoded = decoder.test_forward(distored)
                        decoder_acc = (decoded >= 0).eq(msg >= 0).sum().float() / msg.numel()
                        all_results[att][str(ratio)]['avgs'][0] += decoder_acc.item()
                        all_results[att][str(ratio)]['avgs'][1] += snr_score

                        att_snr, att_pesq, att_utterance_sim, att_spk_sim = fidelity(encoded.detach(), distored.detach(), process_config["audio"]["sample_rate"])
                        all_results[att][str(ratio)]['avgs'][2] += att_snr
                        all_results[att][str(ratio)]['avgs'][3] += att_pesq
                        all_results[att][str(ratio)]['avgs'][4] += att_utterance_sim
                        all_results[att][str(ratio)]['avgs'][5] += att_spk_sim
                        all_results[att][str(ratio)]['f'].write("audio:{},\tsnr:{:.8f}, acc:{:.8f}, att_snr:{:.8f}, att_pesq:{:.8f}, att_utterance_sim:{:.8f}, att_spk_sim:{:.8f}\n".format(name, snr_score.item(), decoder_acc, att_snr, att_pesq, att_utterance_sim, att_spk_sim))
                        if global_step == 1:
                            soundfile.write(all_results[att][str(ratio)]['example'], distored.cpu().squeeze(0).squeeze(0).detach().numpy(), samplerate=process_config["audio"]["sample_rate"])
                        logging.info('-' * 100)
                        logging.info("step:{} - acc:{:.8f} - snr:{:.8f} - att_snr:{:.8f} - att_pesq:{:.8f} - att_utterance_sim:{:.8f} - att_spk_sim:{:.8f} - name:{}".format( \
                            global_step, decoder_acc, snr_score, att_snr, att_pesq, att_utterance_sim, att_spk_sim, sample["name"][0]))
                elif att in [25]: # crop-mel-position
                    for ratio in range(1, 6):
                        distored = dl(encoded, att, ratio)
                        decoded = decoder.mel_test_forward(distored)
                        decoder_acc = (decoded >= 0).eq(msg >= 0).sum().float() / msg.numel()
                        all_results[att][str(ratio)]['avgs'][0] += decoder_acc.item()
                        all_results[att][str(ratio)]['avgs'][1] += snr_score

                        att_snr, att_pesq, att_utterance_sim, att_spk_sim = 0,0,0,0
                        all_results[att][str(ratio)]['avgs'][2] += att_snr
                        all_results[att][str(ratio)]['avgs'][3] += att_pesq
                        all_results[att][str(ratio)]['avgs'][4] += att_utterance_sim
                        all_results[att][str(ratio)]['avgs'][5] += att_spk_sim
                        all_results[att][str(ratio)]['f'].write("audio:{},\tsnr:{:.8f}, acc:{:.8f}, att_snr:{:.8f}, att_pesq:{:.8f}, att_utterance_sim:{:.8f}, att_spk_sim:{:.8f}\n".format(name, snr_score.item(), decoder_acc, att_snr, att_pesq, att_utterance_sim, att_spk_sim))
                        # if global_step == 1:
                        #     soundfile.write(all_results[att][str(ratio)]['example'], distored.cpu().squeeze(0).squeeze(0).detach().numpy(), samplerate=process_config["audio"]["sample_rate"])
                        logging.info('-' * 100)
                        logging.info("step:{} - acc:{:.8f} - snr:{:.8f} - att_snr:{:.8f} - att_pesq:{:.8f} - att_utterance_sim:{:.8f} - att_spk_sim:{:.8f} - name:{}".format( \
                            global_step, decoder_acc, snr_score, att_snr, att_pesq, att_utterance_sim, att_spk_sim, sample["name"][0]))
                elif att in [26]: # crop-mel-wave-position
                    for ratio in range(1, 6):
                        distored = dl(encoded, att, ratio)
                        decoded = decoder.test_forward(distored)
                        decoder_acc = (decoded >= 0).eq(msg >= 0).sum().float() / msg.numel()
                        all_results[att][str(ratio)]['avgs'][0] += decoder_acc.item()
                        all_results[att][str(ratio)]['avgs'][1] += snr_score

                        att_snr, att_pesq, att_utterance_sim, att_spk_sim = fidelity(encoded.detach(), distored.detach(), process_config["audio"]["sample_rate"])
                        all_results[att][str(ratio)]['avgs'][2] += att_snr
                        all_results[att][str(ratio)]['avgs'][3] += att_pesq
                        all_results[att][str(ratio)]['avgs'][4] += att_utterance_sim
                        all_results[att][str(ratio)]['avgs'][5] += att_spk_sim
                        all_results[att][str(ratio)]['f'].write("audio:{},\tsnr:{:.8f}, acc:{:.8f}, att_snr:{:.8f}, att_pesq:{:.8f}, att_utterance_sim:{:.8f}, att_spk_sim:{:.8f}\n".format(name, snr_score.item(), decoder_acc, att_snr, att_pesq, att_utterance_sim, att_spk_sim))
                        if global_step == 1:
                            soundfile.write(all_results[att][str(ratio)]['example'], distored.cpu().squeeze(0).squeeze(0).detach().numpy(), samplerate=process_config["audio"]["sample_rate"])
                        logging.info('-' * 100)
                        logging.info("step:{} - acc:{:.8f} - snr:{:.8f} - att_snr:{:.8f} - att_pesq:{:.8f} - att_utterance_sim:{:.8f} - att_spk_sim:{:.8f} - name:{}".format( \
                            global_step, decoder_acc, snr_score, att_snr, att_pesq, att_utterance_sim, att_spk_sim, sample["name"][0]))
                elif att < 4 or att in [16, 17, 18, 19, 20]:
                    pass
                else:
                    raise("Not implementation error")
        rf.write("{} audios, avg_snr:{:.8f}, avg_pesq:{:.8f}, utterance_sim:{:.8f}, spk_sim:{:.8f}".format(global_step, avg_snr/global_step, avg_pesq/global_step, avg_utterance_sim/global_step, avg_spk_sim/global_step))
        rf.close()
        wmf.close()
    
    # close txt
    for att in range(start, end):
        if att in [4,5,6]:
            for ratio in [20,40,60,80,90,95,96,97,98]:
                temp = np.array(all_results[att][str(ratio)]['avgs'])/global_step
                all_results[att][str(ratio)]['f'].write("{} audios, avg_acc:{:.8f}, avg_snr:{:.8f}, avg_att_snr:{:.8f}, avg_pesq:{:.8f}, utterance_sim:{:.8f}, spk_sim:{:.8f}\n".format(global_step, temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]))
                all_results[att][str(ratio)]['f'].write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t".format(temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]))
                all_results[att][str(ratio)]['f'].close()
        elif att in [10]:
            for ratio in [20,40,60,80]:
                temp = np.array(all_results[att][str(ratio)]['avgs'])/global_step
                all_results[att][str(ratio)]['f'].write("{} audios, avg_acc:{:.8f}, avg_snr:{:.8f}, avg_att_snr:{:.8f}, avg_pesq:{:.8f}, utterance_sim:{:.8f}, spk_sim:{:.8f}\n".format(global_step, temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]))
                all_results[att][str(ratio)]['f'].write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t".format(temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]))
                all_results[att][str(ratio)]['f'].close()
        elif att in [9]: # gn
            for ratio in [10,20,30,40,50,60,70,80]:
                temp = np.array(all_results[att][str(ratio)]['avgs'])/global_step
                all_results[att][str(ratio)]['f'].write("{} audios, avg_acc:{:.8f}, avg_snr:{:.8f}, avg_att_snr:{:.8f}, avg_pesq:{:.8f}, utterance_sim:{:.8f}, spk_sim:{:.8f}\n".format(global_step, temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]))
                all_results[att][str(ratio)]['f'].write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t".format(temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]))
                all_results[att][str(ratio)]['f'].close()
        elif att in [11]: #mp3
            for ratio in [8,16,24,32,40,48,56,64]:
                temp = np.array(all_results[att][str(ratio)]['avgs'])/global_step
                all_results[att][str(ratio)]['f'].write("{} audios, avg_acc:{:.8f}, avg_snr:{:.8f}, avg_att_snr:{:.8f}, avg_pesq:{:.8f}, utterance_sim:{:.8f}, spk_sim:{:.8f}\n".format(global_step, temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]))
                all_results[att][str(ratio)]['f'].write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t".format(temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]))
                all_results[att][str(ratio)]['f'].close()
        elif att in [13]: #medianfilt
            for ratio in [5, 15, 25, 35]:
                temp = np.array(all_results[att][str(ratio)]['avgs'])/global_step
                all_results[att][str(ratio)]['f'].write("{} audios, avg_acc:{:.8f}, avg_snr:{:.8f}, avg_att_snr:{:.8f}, avg_pesq:{:.8f}, utterance_sim:{:.8f}, spk_sim:{:.8f}\n".format(global_step, temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]))
                all_results[att][str(ratio)]['f'].write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t".format(temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]))
                all_results[att][str(ratio)]['f'].close()
        elif att in [7,8,12,14,15]:
            ratio = 0
            temp = np.array(all_results[att][str(ratio)]['avgs'])/global_step
            all_results[att][str(ratio)]['f'].write("{} audios, avg_snr:{:.8f}, avg_acc:{:.8f}, avg_att_snr:{:.8f}, avg_pesq:{:.8f}, utterance_sim:{:.8f}, spk_sim:{:.8f}\n".format(global_step, temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]))
            all_results[att][str(ratio)]['f'].write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t".format(temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]))
            all_results[att][str(ratio)]['f'].close()
        # elif att in [17, 18]: # crop-mel
        #     for ratio in [10,20,30,40,50,60,70,80]:
        #         temp = np.array(all_results[att][str(ratio)]['avgs'])/global_step
        #         all_results[att][str(ratio)]['f'].write("{} audios, avg_acc:{:.8f}, avg_snr:{:.8f}, avg_att_snr:{:.8f}, avg_pesq:{:.8f}, utterance_sim:{:.8f}, spk_sim:{:.8f}\n".format(global_step, temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]))
        #         all_results[att][str(ratio)]['f'].write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t".format(temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]))
        #         all_results[att][str(ratio)]['f'].close()
        # elif att in [19,20]: # crop-mel-wave
        #     for ratio in [10,20,30,40,50,60,70,80]:
        #         temp = np.array(all_results[att][str(ratio)]['avgs'])/global_step
        #         all_results[att][str(ratio)]['f'].write("{} audios, avg_acc:{:.8f}, avg_snr:{:.8f}, avg_att_snr:{:.8f}, avg_pesq:{:.8f}, utterance_sim:{:.8f}, spk_sim:{:.8f}\n".format(global_step, temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]))
        #         all_results[att][str(ratio)]['f'].write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]))
        #         all_results[att][str(ratio)]['f'].close()
        elif att in [21]: # crop-mel-position
            for ratio in [1,2,3,4,5,6,7,8,9,10]:
                temp = np.array(all_results[att][str(ratio)]['avgs'])/global_step
                all_results[att][str(ratio)]['f'].write("{} audios, avg_acc:{:.8f}, avg_snr:{:.8f}, avg_att_snr:{:.8f}, avg_pesq:{:.8f}, utterance_sim:{:.8f}, spk_sim:{:.8f}\n".format(global_step, temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]))
                all_results[att][str(ratio)]['f'].write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t".format(temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]))
                all_results[att][str(ratio)]['f'].close()
        elif att in [22]: # crop-mel-wave-position
            for ratio in [1,2,3,4,5,6,7,8,9,10]:
                temp = np.array(all_results[att][str(ratio)]['avgs'])/global_step
                all_results[att][str(ratio)]['f'].write("{} audios, avg_acc:{:.8f}, avg_snr:{:.8f}, avg_att_snr:{:.8f}, avg_pesq:{:.8f}, utterance_sim:{:.8f}, spk_sim:{:.8f}\n".format(global_step, temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]))
                all_results[att][str(ratio)]['f'].write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]))
                all_results[att][str(ratio)]['f'].close()

        elif att in [23]: # crop-mel-position
            for ratio in range(1, 21):
                temp = np.array(all_results[att][str(ratio)]['avgs'])/global_step
                all_results[att][str(ratio)]['f'].write("{} audios, avg_acc:{:.8f}, avg_snr:{:.8f}, avg_att_snr:{:.8f}, avg_pesq:{:.8f}, utterance_sim:{:.8f}, spk_sim:{:.8f}\n".format(global_step, temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]))
                all_results[att][str(ratio)]['f'].write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t".format(temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]))
                all_results[att][str(ratio)]['f'].close()
        elif att in [24]: # crop-mel-wave-position
            for ratio in range(1, 21):
                temp = np.array(all_results[att][str(ratio)]['avgs'])/global_step
                all_results[att][str(ratio)]['f'].write("{} audios, avg_acc:{:.8f}, avg_snr:{:.8f}, avg_att_snr:{:.8f}, avg_pesq:{:.8f}, utterance_sim:{:.8f}, spk_sim:{:.8f}\n".format(global_step, temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]))
                all_results[att][str(ratio)]['f'].write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]))
                all_results[att][str(ratio)]['f'].close()
        elif att in [25]: # crop-mel-position
            for ratio in range(1, 6):
                temp = np.array(all_results[att][str(ratio)]['avgs'])/global_step
                all_results[att][str(ratio)]['f'].write("{} audios, avg_acc:{:.8f}, avg_snr:{:.8f}, avg_att_snr:{:.8f}, avg_pesq:{:.8f}, utterance_sim:{:.8f}, spk_sim:{:.8f}\n".format(global_step, temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]))
                all_results[att][str(ratio)]['f'].write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t".format(temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]))
                all_results[att][str(ratio)]['f'].close()
        elif att in [26]: # crop-mel-wave-position
            for ratio in range(1, 6):
                temp = np.array(all_results[att][str(ratio)]['avgs'])/global_step
                all_results[att][str(ratio)]['f'].write("{} audios, avg_acc:{:.8f}, avg_snr:{:.8f}, avg_att_snr:{:.8f}, avg_pesq:{:.8f}, utterance_sim:{:.8f}, spk_sim:{:.8f}\n".format(global_step, temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]))
                all_results[att][str(ratio)]['f'].write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]))
                all_results[att][str(ratio)]['f'].close()
        # elif att in [23,24,25]:
        #     for ratio in [99]:
        #         temp = np.array(all_results[att][str(ratio)]['avgs'])/global_step
        #         all_results[att][str(ratio)]['f'].write("{} audios, avg_acc:{:.8f}, avg_snr:{:.8f}, avg_att_snr:{:.8f}, avg_pesq:{:.8f}, utterance_sim:{:.8f}, spk_sim:{:.8f}\n".format(global_step, temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]))
        #         all_results[att][str(ratio)]['f'].write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t".format(temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]))
        #         all_results[att][str(ratio)]['f'].close()
        elif att < 4 or att in [16, 17, 18, 19, 20]:
            pass
        else:
            raise("Not implementation error")
    



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
