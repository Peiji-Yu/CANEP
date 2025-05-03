import argparse
import utils
import sys
import os
import datetime
import time
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import conttime
import module
import torch
import matplotlib.pyplot as plt
import test
import torch.nn.functional as F
import torch.nn as nn
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:9216"
print(torch.cuda.device_count())
torch.manual_seed(42)
import tqdm
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training model..")

    parser.add_argument("--dataset", type=str, help="e.g. hawkes", required=True)
    parser.add_argument("--model", type=str, help="e.g. hawkes", required=True)
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="maximum epochs")
    parser.add_argument("--seq_len", type=int, default=-1, help="truncated sequence length for hawkes and self-correcting, -1 means full sequence")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch_size for each train iteration")
    parser.add_argument("--used_past_model", type=bool, help="True to use a trained model named model.pt")
    parser.add_argument('--embedding_size', type=int, default=10)
    config = parser.parse_args()

    dataset = config.dataset
    lr = config.lr
    seq_len = config.seq_len
    num_epochs = config.epochs
    batch_size = config.batch_size
    used_model = config.used_past_model
    model_name = config.model
    embedding_size = config.embedding_size
    now = str(datetime.datetime.today()).split()
    now = now[0]+"-"+now[1][:5]
    id_process = os.getpid()
    print("id: " + str(id_process))
    if not os.path.exists('log'):
        os.makedirs('log')
    log_file_name = "log/train_process"+str(id_process)+".txt"
    log = open(log_file_name, 'w')
    log.write("Data when training: " + str(datetime.datetime.now()))
    log.write("\nTraining-id: " + str(id_process))
    log.write("\nTraining data: " + dataset)
    log.write("\nLearning rate: " + str(lr))
    log.write("\nMax epochs: " + str(num_epochs))
    log.write("\nseq lens: " + str(seq_len))
    log.write("\nbatch size for train: " + str(batch_size))
    log.write("\nuse previous model: " + str(used_model))
    log.write("\nmodel:"+model_name)
    log.write("\nembedding_size:" + str(embedding_size))
    t1 = time.time()
    print("Processing data...")
    if dataset == 'hawkes' or dataset == "self-correcting":
        file_path = 'data/' + dataset + "/time-train.txt"   # train file
        test_path = 'data/' + dataset + '/time-test.txt'    # test file
        time_duration, seq_lens_list = utils.open_txt_file(file_path)   # train time info
        test_duration, seq_lens_test = utils.open_txt_file(test_path)   # test time info
        event_size = 1
        event_train = utils.get_index_txt(time_duration) # train type
        type_test = utils.get_index_txt(test_duration)  # test type
        if seq_len == -1:
            time_duration, event_train = utils.padding_full(time_duration, event_train, seq_lens_list, event_size)
            test_duration, type_test = utils.padding_full(test_duration, type_test, seq_lens_test, event_size)
        else:
            time_duration, event_train, seq_lens_list = utils.padding_seq_len(time_duration, event_train, event_size, seq_len)
            test_duration, type_test = utils.padding_seq_len(test_duration, type_test, event_size, seq_len)
    else:
        if dataset == 'conttime' or dataset == "data_hawkes" or dataset == "data_hawkeshib":
            event_size = 8
        elif dataset == 'data_mimic1' or dataset == 'data_mimic2' or dataset == 'data_mimic3' or dataset == 'data_mimic4' or\
        dataset == 'data_mimic5':
            event_size = 75
        elif dataset == 'data_so1' or dataset == 'data_so2' or dataset == 'data_so3' or dataset == 'data_so4' or\
        dataset == 'data_so5':
            event_size = 22
        elif dataset == 'data_book1' or dataset == 'data_book2' or dataset == 'data_book3' or dataset == 'data_book4'\
        or dataset == 'data_book5':
            event_size = 3
        elif dataset == 'ml-10M' or dataset == 'ml-10M100K' or dataset == 'ml-small':
            event_size = 1000
            if model_name == 'CEHawkes':
                type_size = 308
        elif dataset == 'ml':
            event_size = 1000
            if model_name == 'CEHawkes':
                type_size = 20
        elif dataset == 'data_game' or dataset == 'data_gameCE':
            event_size = 1050
            if model_name == 'CEHawkes':
                type_size = 35
        elif dataset == 'lastfm':
            event_size = 1000
            if model_name == 'CEHawkes':
                type_size = 254
        else:
            print("Data process file for other types of datasets have not been developed yet, or the dataset is not found")
            log.write("\nData process file for other types of datasets have not been developed yet, or the datase is not found")
            log.close()
            sys.exit()
        file_path = 'data/' + dataset + '/train.pkl'
        test_path = 'data/' + dataset + '/dev.pkl'
        type_train = []
        type_test = []
        if model_name == 'Hawkes' or model_name == "EHawkes":
            time_duration, event_train, seq_lens_list = utils.open_pkl_file(file_path, 'train')
            test_duration, event_test, seq_lens_test = utils.open_pkl_file(test_path, 'dev')
        else:
            time_duration, event_train, type_train,seq_lens_list = utils.open_pkl_file2(file_path, 'train')
            test_duration, event_test, type_test, seq_lens_test = utils.open_pkl_file2(test_path, 'dev')

        if dataset =='ml-10M' or dataset == 'ml-10M100K' or dataset == 'ml-small' or dataset=='data_gameCE' or dataset=='data_game' or dataset == 'ml' or dataset == 'lastfm':
            time_duration, event_train, type_train = utils.padding_full2(time_duration, event_train, type_train, model_name)
            test_duration, type_test, type_test = utils.padding_full2(test_duration, event_test, type_test, model_name)
        else:
            time_duration, event_train = utils.padding_full(time_duration, event_train, seq_lens_list, event_size)
            test_duration, type_test = utils.padding_full(test_duration, event_test, seq_lens_test, event_size)

        # time_duration, events, event_size = utils.open_pkl_file3(file_path, 'train')
        # test_duration, type_test, event_size = utils.open_pkl_file3(test_path, 'dev')
        # time_duration, events = utils.padding_full2(time_duration, events, event_size)
        # test_duration, type_test = utils.padding_full2(test_duration, type_test, event_size)


    train_data = utils.Data_Batch(time_duration, event_train, type_train, seq_lens_list, model_name)
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    print("Data Processing Finished...")
    t2 = time.time()
    data_process_time = t2 - t1
    print("Getting data takes: " + str(data_process_time) + " seconds")
    log.write("\n\nGetting data takes: " + str(data_process_time) + " seconds")

    print("start training...")
    t3 = time.time()
    if used_model:
        model = torch.load("{model.pt")
    else:
        # model = conttime.Conttime(n_types=event_size, lr=lr)
        if model_name == 'Hawkes':
            model = module.HawkesProcessModel(event_size)
        elif model_name == 'EHawkes':
            model = module.EHawkesProcessModel(event_size, embedding_size, lr)
        else:
            model = module.CEHawkesProcessModel(event_size, type_size, embedding_size, lr)


    # model = nn.DataParallel(model, device_ids=[0, 1])
    if torch.cuda.is_available(): 
        device = torch.device('cuda')
        print("You are using GPU acceleration.")
        log.write("\nYou are using GPU acceleration.")
        print("Number of GPU: ", torch.get_num_threads())
        log.write("\n\nNumber of GPU: " + str((torch.get_num_threads())))
    else:
        device = torch.device("cpu")
        print("CUDA is not Available. You are using CPU only.")
        log.write("\nCUDA is not Available. You are using CPU only.")
        print("Number of cores: ", os.cpu_count())
        log.write("\n\nNumber of cores: " + str(os.cpu_count()))

    loss_value = []
    model = model.to(device)
    log_test_list = []
    log_time_list = []
    log_type_list = []

    type_accuracy_list = []
    for i in range(num_epochs):
        loss_total = 0
        events_total = 0
        max_len = len(train_data)
        loss_list = []
        count = 0
        for idx, a_batch in enumerate(tqdm.tqdm(train_data)):
            durations, event_items, seq_lens = a_batch['duration_seq'], a_batch['event_seq'], a_batch['seq_len']
            event_items = event_items.to(device)
            durations = durations.to(device)
            if model_name == 'CEHawkes':
                type_items = a_batch['type_seq'].to(device)

            # sim_durations, total_time_seqs, time_simulation_index = utils.generate_simulation(durations, seq_lens)
            event_items = event_items.to(device)
            durations = durations.to(device)
            # sim_durations.to(device)
            # total_time_seqs.to(device)
            # time_simulation_index.to(device)
            if model_name != 'CEHawkes':
                batch = (event_items, durations)
                loss = model.train_batch(batch, 1)
            else:
                batch = (event_items, type_items, durations)
                loss = model.train_batch(batch, 1)
            if loss==torch.nan or loss==torch.inf:
                continue
            count += 1
            if model_name == 'Hawkes':
                log_likelihood = -loss
            else:
                log_likelihood = -loss.item()
            total_size = torch.sum(seq_lens)
            loss_list.append(log_likelihood)
            loss_total += log_likelihood
            events_total += total_size
            # print("In epochs {0}, process {1} over {2} is done".format(i, idx, max_len))
        # avg_log = loss_total / events_total
        print(count)
        # loss_value.append(-avg_log)
        print("The log-likelihood at epochs {0} is {1}".format(i, loss_total))
        log.write("\nThe log likelihood at epochs {0} is {1}".format(i, loss_total))
        print("model saved..")
        # if not os.path.exists("model_weight"):
        # #     os.mkdir("model_weight")
        torch.save(model.state_dict(), f"/model/hawkes/{model_name}-{dataset}-{str(id_process)}-model.pth") 
    print("Training Finished...")
    print(id_process)




