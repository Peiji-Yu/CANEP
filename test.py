import numpy as np
import argparse
import module
import utils
import sys
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def evaluate_metrics(predictions, targets, k_list=[1, 5, 10, 20]):
    """
    计算Recall@K和MRR指标
    :param predictions: 预测概率矩阵 [n_samples, n_classes]
    :param targets: 真实标签 [n_samples]
    :param k_list: 要计算的K值列表
    :return: 包含各项指标的字典
    """
    results = {}

    # 计算每个样本的排名
    ranked_indices = np.argsort(-predictions, axis=1)
    target_ranks = np.where(ranked_indices == targets.reshape(-1, 1))[1] + 1

    # 计算Recall@K
    for k in k_list:
        recall_at_k = np.mean(target_ranks <= k)
        results[f'recall@{k}'] = recall_at_k

    # 计算MRR
    for k in k_list:
        reciprocal_ranks = np.where(target_ranks>k, 0, 1.0 / target_ranks)
        mrr = np.mean(reciprocal_ranks)
        results[f'mrr@{k}'] = mrr

    return results
def test_model(model, test_loader, model_name, T, device):
    """
    测试模型性能
    :param model: 训练好的模型
    :param test_loader: 测试数据加载器
    :param T: 观察时间窗口
    :param device: 计算设备
    :return: 评估指标字典
    """
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            # 获取测试数据
            if model_name != 'CEHawkes':
                event_times, event_seq = batch['duration_seq'], batch['event_seq']
            else:
                event_times, event_seq, category_seq = batch['duration_seq'], batch['event_seq'], batch['type_seq']
            event_times = event_times.to(torch.float32).to(device)
            event_seq = event_seq.long().to(device)
            if model_name == 'CEHawkes':
                category_seq = category_seq.long().to(device)

            batch_size, _ = event_seq.shape

            # 对每个序列中的每个事件进行预测
            for b in range(batch_size):
                seq_len = len(event_seq[b])
                for i in range(1, seq_len):  # 从第二个事件开始预测
                    # 使用历史事件预测当前事件
                    if model_name == 'CEHawkes':
                        intensities = model.hawkes_process.compute_intensity(
                            event_times[b, :i],
                            event_seq[b, :i],
                            category_seq[b, :i],
                            event_times[b, i]
                        )
                    else:
                        intensities = model.hawkes_process.compute_intensity(
                            event_times[b, :i],
                            event_seq[b, :i],
                            event_times[b, i]
                        )
                    probs = F.softmax(intensities, dim=0)

                    # 保存预测和真实值
                    all_predictions.append(probs.cpu().numpy())
                    all_targets.append(event_seq[b, i].item())

    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # 计算指标
    metrics = evaluate_metrics(all_predictions, all_targets)

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str,
                        help="e.g hawkes. Must be the same as the training data in training model")
    parser.add_argument("--model", type=str, default='EHawkes',
                        help="e.g hawkes. Must be the same as the training data in training model")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch_size for each train iteration")
    parser.add_argument("--model_path", type=str, default="model_weight/EHawkes-2773-model.pth", help="The path of trained model")
    parser.add_argument("--n_samples", type=int, help="number of samples of monte carlo integration in test 1",
                        default=10000)
    parser.add_argument("--embedding_size", type=int, default=32)
    
    id_process = os.getpid()
    config = parser.parse_args()
    dataset = config.dataset
    model_name = config.model
    batch_size = config.batch_size
    model_path = config.model_path
    n_samples = config.n_samples
    embedding_size = config.embedding_size
    print("Testing...")
    log_file_name = "test_log/test_process" + str(id_process) + ".txt"
    if not os.path.exists('test_log'):
        os.makedirs('test_log')
    log = open(log_file_name, 'w')

    log.write("\nTest dataset: " + dataset)

    print("processing testing data set")
    type_test = []
    if dataset == 'hawkes' or dataset == "self-correcting":
        file_path = 'data/' + dataset + "/time-test.txt"
        test_duration, seq_lens_list = utils.open_txt_file(file_path)
        event_size = 1
        event_test = utils.get_index_txt(test_duration)
        test_duration, event_test = utils.padding_full(test_duration, event_test, seq_lens_list, event_size)
    elif dataset == 'ml-10M' or dataset == 'ml-10M100K' or dataset == 'ml-small':
        file_path = 'data/' + dataset + "/test.pkl"
        event_size = 1000
        if model_name == 'Hawkes' or model_name == "EHawkes":
            test_duration, event_test, seq_lens_list = utils.open_pkl_file(file_path, 'test')
        else:
            type_size = 308
            test_duration, event_test, type_test, seq_lens_list = utils.open_pkl_file2(file_path, 'test')
        test_duration, event_test, type_test = utils.padding_full2(test_duration, event_test, seq_lens_list, event_size)
    elif dataset == 'data_gameCE' or dataset == 'data_game':
        file_path = 'data/' + dataset + "/test.pkl"
        event_size = 1050
        if model_name == 'Hawkes' or model_name == "EHawkes":
            test_duration, event_test, seq_lens_list = utils.open_pkl_file(file_path, 'test')
        else:
            type_size = 35
            test_duration, event_test, type_test, seq_lens_list = utils.open_pkl_file2(file_path, 'test')
        test_duration, event_test, type_test = utils.padding_full2(test_duration, event_test, seq_lens_list, event_size)    
    elif dataset == 'lastfm':
        file_path = 'data/' + dataset + "/test.pkl"
        event_size = 1000
        if model_name == 'Hawkes' or model_name == "EHawkes":
            test_duration, event_test, seq_lens_list = utils.open_pkl_file(file_path, 'test')
        else:
            type_size = 254
            test_duration, event_test, type_test, seq_lens_list = utils.open_pkl_file2(file_path, 'test')
        test_duration, event_test, type_test = utils.padding_full2(test_duration, event_test, type_test, model_name) 
        
    else:
        print("Data process file for other types of datasets have not been developed yet, or the dataset is not found")
        log.close()
        sys.exit()
    log.write(f'\ndataset:{dataset}\nmodel_name:{model_name}\n')
    test_data = utils.Data_Batch(test_duration, event_test, type_test, seq_lens_list, model_name)
    test_data = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    print("testing dataset is done")
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


    if model_name == 'EHawkes':
        model = module.EHawkesProcessModel(event_size, embedding_size=embedding_size).to(device)
    elif model_name == 'CEHawkes':
        model = module.CEHawkesProcessModel(event_size, type_size, embedding_size).to(device)
    else:
        model = module.HawkesProcessModel(event_size).to(device)
    model.load_state_dict(torch.load(model_path))
    test_metrics = test_model(model, test_data, model_name, T=1.0, device=device)
    print("Test Metrics:")
    log.write("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
        log.write(f"\n{metric}: {value:.4f}")
    log.close()
