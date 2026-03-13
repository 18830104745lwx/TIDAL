import os
import argparse

from libcity.pipeline import run_model
from libcity.utils import general_arguments, str2bool, str2float


def add_other_args(parser):
    for arg in general_arguments:
        if general_arguments[arg] == 'int':
            parser.add_argument('--{}'.format(arg), type=int, default=None)
        elif general_arguments[arg] == 'bool':
            parser.add_argument('--{}'.format(arg),
                                type=str2bool, default=None)
        elif general_arguments[arg] == 'str':
            parser.add_argument('--{}'.format(arg),
                                type=str, default=None)
        elif general_arguments[arg] == 'float':
            parser.add_argument('--{}'.format(arg),
                                type=str2float, default=None)
        elif general_arguments[arg] == 'list of int':
            parser.add_argument('--{}'.format(arg), nargs='+',
                                type=int, default=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str,
                        default='traffic_state_pred', help='the name of task')
    parser.add_argument('--model', type=str,
                        default='GRU', help='the name of model')
    parser.add_argument('--dataset', type=str,
                        default=None, help='the name of dataset (overrides config_file)')
    parser.add_argument('--config_file', type=str,
                        default=None, help='the file name of config file')
    parser.add_argument('--saved_model', type=str2bool,
                        default=True, help='whether save the trained model')
    parser.add_argument('--train', type=str2bool, default=True,
                        help='whether re-train model if the model is \
                             trained before')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--exp_id', type=str,
                        default=None, help='id of experiment')
    add_other_args(parser)
    args = parser.parse_args()

    # 如果未指定 --dataset，从 config_file 中读取
    if args.dataset is None and args.config_file is not None:
        import json
        config_path = './{}.json'.format(args.config_file)
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            args.dataset = cfg.get('dataset', 'ShanghaiTaxi20x20')
        else:
            args.dataset = 'ShanghaiTaxi20x20'
    elif args.dataset is None:
        args.dataset = 'ShanghaiTaxi20x20'

    dict_args = vars(args)
    other_args = {key: val for key, val in dict_args.items() if key not in [
        'task', 'model', 'dataset', 'config_file', 'saved_model', 'train'] and
        val is not None}
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpu_id))
    run_model(task=args.task, model_name=args.model, dataset_name=args.dataset,
              config_file=args.config_file, saved_model=args.saved_model,
              train=args.train, other_args=other_args)
