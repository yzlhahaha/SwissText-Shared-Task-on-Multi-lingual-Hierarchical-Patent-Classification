import torch
import time
import warnings
from pathlib import Path
from argparse import ArgumentParser
from pybert.train.losses import BCEWithLogLoss
from pybert.train.trainer import Trainer
from torch.utils.data import DataLoader
from pybert.io.utils import collate_fn
from pybert.io.bert_processor import BertProcessor
from pybert.common.tools import init_logger, logger
from pybert.common.tools import seed_everything
from pybert.configs.basic_config import config
from pybert.model.bert_for_multi_label import BertForMultiLable
# from pybert.preprocessing.preprocessor import EnglishPreProcessor
from pybert.callback.modelcheckpoint import ModelCheckpoint
from pybert.callback.trainingmonitor import TrainingMonitor
from pybert.train.metrics import AUC, AccuracyThresh, MultiLabelReport, F1Score
from pybert.callback.optimizater.adamw import AdamW
from pybert.callback.lr_schedulers import get_linear_schedule_with_warmup
from torch.utils.data import RandomSampler, SequentialSampler
import os
import json
import numpy as np
import csv

warnings.filterwarnings("ignore")

def get_valid_dataloader(args):
    # --------- data
    # processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=args.do_lower_case)
    processor = BertProcessor()
    # label_list = processor.get_labels()
    # label2id = {label: i for i, label in enumerate(label_list)}
    # id2label = {i: label for i, label in enumerate(label_list)}

    train_data = processor.get_train(config['data_dir'] / f"all_valid.valid.pkl")
    train_examples = processor.create_examples(lines=train_data,
                                               example_type='train',
                                               cached_examples_file=config[
                                                    'cached_dir'] / f"cached_all_valid_examples_{args.arch}")
    train_features = processor.create_features(examples=train_examples,
                                               max_seq_len=args.train_max_seq_len,
                                               cached_features_file=config[
                                                    'cached_dir'] / "cached_all_valid_features_{}_{}".format(
                                                    args.train_max_seq_len, args.arch
                                               ))
    train_dataset = processor.create_dataset(train_features, is_sorted=args.sorted)
    if args.sorted:
        train_sampler = SequentialSampler(train_dataset)
    else:
        train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)
    return train_dataloader

def get_dataloader(args, data_name):
    # --------- data
    # processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=args.do_lower_case)
    processor = BertProcessor()
    # label_list = processor.get_labels()
    # label2id = {label: i for i, label in enumerate(label_list)}
    # id2label = {i: label for i, label in enumerate(label_list)}

    train_data = processor.get_train(config['data_dir'] / f"{data_name}.train.pkl")
    train_examples = processor.create_examples(lines=train_data,
                                               example_type='train',
                                               cached_examples_file=config[
                                                    'cached_dir'] / f"cached_train_examples_{data_name}_{args.arch}")
    train_features = processor.create_features(examples=train_examples,
                                               max_seq_len=args.train_max_seq_len,
                                               cached_features_file=config[
                                                    'cached_dir'] / "cached_train_features_{}_{}_{}".format(
                                                   data_name, args.train_max_seq_len, args.arch
                                               ))
    train_dataset = processor.create_dataset(train_features, is_sorted=args.sorted)
    if args.sorted:
        train_sampler = SequentialSampler(train_dataset)
    else:
        train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)
    return train_dataloader

    # valid_data = processor.get_dev(config['data_dir'] / f"{data_name}.valid.pkl")
    # valid_examples = processor.create_examples(lines=valid_data,
    #                                            example_type='valid',
    #                                            cached_examples_file=config[
    #                                             'data_dir'] / f"cached_valid_examples_{data_name}_{args.arch}")
    #
    # valid_features = processor.create_features(examples=valid_examples,
    #                                            max_seq_len=args.eval_max_seq_len,
    #                                            cached_features_file=config[
    #                                             'data_dir'] / "cached_valid_features_{}_{}_{}".format(
    #                                                data_name, args.eval_max_seq_len, args.arch
    #                                            ))
    # valid_dataset = processor.create_dataset(valid_features)
    # valid_sampler = SequentialSampler(valid_dataset)
    # valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.eval_batch_size,
    #                               collate_fn=collate_fn)
    # return train_dataloader, valid_dataloader

def run_train(args, data_names):
    # --------- data
    # processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=args.do_lower_case)
    processor = BertProcessor()
    label_list = processor.get_labels()
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    # train_data = processor.get_train(config['data_dir'] / f"{data_name}.train.pkl")
    # train_examples = processor.create_examples(lines=train_data,
    #                                            example_type='train',
    #                                            cached_examples_file=config[
    #                                                 'data_dir'] / f"cached_train_examples_{args.arch}")
    # train_features = processor.create_features(examples=train_examples,
    #                                            max_seq_len=args.train_max_seq_len,
    #                                            cached_features_file=config[
    #                                                 'data_dir'] / "cached_train_features_{}_{}".format(
    #                                                args.train_max_seq_len, args.arch
    #                                            ))
    # train_dataset = processor.create_dataset(train_features, is_sorted=args.sorted)
    # if args.sorted:
    #     train_sampler = SequentialSampler(train_dataset)
    # else:
    #     train_sampler = RandomSampler(train_dataset)
    # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
    #                               collate_fn=collate_fn)
    #
    # valid_data = processor.get_dev(config['data_dir'] / f"{data_name}.valid.pkl")
    # valid_examples = processor.create_examples(lines=valid_data,
    #                                            example_type='valid',
    #                                            cached_examples_file=config[
    #                                             'data_dir'] / f"cached_valid_examples_{args.arch}")
    #
    # valid_features = processor.create_features(examples=valid_examples,
    #                                            max_seq_len=args.eval_max_seq_len,
    #                                            cached_features_file=config[
    #                                             'data_dir'] / "cached_valid_features_{}_{}".format(
    #                                                args.eval_max_seq_len, args.arch
    #                                            ))
    # valid_dataset = processor.create_dataset(valid_features)
    # valid_sampler = SequentialSampler(valid_dataset)
    # valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.eval_batch_size,
    #                               collate_fn=collate_fn)

    # ------- model
    logger.info("initializing model")
    if args.resume_path:
        args.resume_path = Path(args.resume_path)
        model = BertForMultiLable.from_pretrained(args.resume_path, num_labels=len(label_list))
    else:
        # model = BertForMultiLable.from_pretrained(config['bert_model_dir'], num_labels=len(label_list))
        model = BertForMultiLable.from_pretrained("bert-base-multilingual-cased", num_labels=len(label_list))
    #t_total = int(len(train_dataloader) / args.gradient_accumulation_steps * args.epochs)
    t_total = 200000
  
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
         {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay': args.weight_decay},
         {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                   num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # ---- callbacks
    logger.info("initializing callbacks")
    train_monitor = TrainingMonitor(file_dir=config['figure_dir'], arch=args.arch)
    model_checkpoint = ModelCheckpoint(checkpoint_dir=config['checkpoint_dir'],mode=args.mode,
                                       monitor=args.monitor,arch=args.arch,
                                       save_best_only=args.save_best)

    # **************************** training model ***********************
    logger.info("***** Running training *****")
    #logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    trainer = Trainer(args= args,model=model,logger=logger,criterion=BCEWithLogLoss(),optimizer=optimizer,
                      scheduler=scheduler,early_stopping=None,training_monitor=train_monitor,
                      model_checkpoint=model_checkpoint,
                      batch_metrics=[AccuracyThresh(thresh=0.5)],
                      epoch_metrics=[AUC(average='micro', task_type='binary'),
                                     MultiLabelReport(id2label=id2label),
                                     F1Score(average='micro', task_type='binary')])

    trainer.model.zero_grad()
    seed_everything(trainer.args.seed)  # Added here for reproductibility (even between python 2 a
    
    iter_num = 0
    valid_dataloader = get_valid_dataloader(args)
    for epoch in range(trainer.start_epoch, trainer.start_epoch + trainer.args.epochs):
        trainer.logger.info(f"Epoch {epoch}/{trainer.args.epochs}")
        update_epoch = True

        for i, data_name in enumerate(data_names):
            filename_int = int(data_name)
            if filename_int > 3500:
                continue
            trainer.logger.info(f"Epoch {epoch} - summary {i+1}/{len(data_names)}"+ f": summary_{data_name}")
            # train_dataloader, valid_dataloader = get_dataloader(args, data_name)
            train_dataloader = get_dataloader(args, data_name)
            # train_log, valid_log = trainer.train(train_data=train_dataloader, valid_data=valid_dataloader, epoch=update_epoch)
            train_log = trainer.train(train_data=train_dataloader, epoch=update_epoch)
            update_epoch = False

            # if train_log == None:
            #     continue
            
            iter_num += 1

            # logs = dict(train_log)
            # show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
            # trainer.logger.info(show_info)


            if iter_num % 50 == 0:
                valid_log = trainer.valid_epoch(valid_dataloader)
                logs = dict(valid_log)
                show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
                trainer.logger.info(show_info)

                # save
                if trainer.training_monitor:
                    trainer.training_monitor.epoch_step(logs)

            # save model
            if trainer.model_checkpoint:
                if iter_num % 50 == 0:
                #     state = trainer.save_info(epoch, best=logs[trainer.model_checkpoint.monitor])
                    state = trainer.save_info(iter_num, best=logs[trainer.model_checkpoint.monitor])
                    trainer.model_checkpoint.bert_epoch_step(current=logs[trainer.model_checkpoint.monitor], state=state)

            # early_stopping
            if trainer.early_stopping:
                trainer.early_stopping.epoch_step(epoch=epoch, current=logs[trainer.early_stopping.monitor])
                if trainer.early_stopping.stop_training:
                    break


def run_test(args):
    from pybert.io.task_data import TaskData
    from pybert.test.predictor import Predictor
    data = TaskData()
    # targets, sentences = data.read_data(raw_data_path=config['test_path'],
    #                                     preprocessor=EnglishPreProcessor(),
    #                                     is_train=False)
    _, _, targets, sentences = data.read_data(config, raw_data_path=config['test_path'],
                                        is_train=False)
    lines = list(zip(sentences, targets))
    # processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=args.do_lower_case)
    processor = BertProcessor()
    label_list = processor.get_labels()
    id2label = {i: label for i, label in enumerate(label_list)}

    test_data = processor.get_test(lines=lines)
    test_examples = processor.create_examples(lines=test_data,
                                              example_type='test',
                                              cached_examples_file=config[
                                            'data_dir'] / f"cached_test_examples_{args.arch}")
    test_features = processor.create_features(examples=test_examples,
                                              max_seq_len=args.eval_max_seq_len,
                                              cached_features_file=config[
                                            'data_dir'] / "cached_test_features_{}_{}".format(
                                                args.eval_max_seq_len, args.arch
                                              ))
    test_dataset = processor.create_dataset(test_features)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.train_batch_size,
                                 collate_fn=collate_fn)
    model = BertForMultiLable.from_pretrained(config['checkpoint_dir'], num_labels=len(label_list))

    # ----------- predicting
    logger.info('model predicting....')
    predictor = Predictor(model=model,
                          logger=logger,
                          n_gpu=args.n_gpu)
    result = predictor.predict(data=test_dataloader)
    result[result<0.5] = 0
    result[result>=0.5] = 1
    labels = []
    for i in range(result.shape[0]):
        ids = np.where(result[i]==1)[0]
        each_patent_label = [id2label[id] for id in ids]
        labels.append(each_patent_label)
    if os.path.exists(config['predictions']):
        os.remove(config['predictions'])
    with open(config['test_path'], 'r') as f:
        reader = csv.reader(f)
        for j, line in enumerate(reader):
            id = line[0]
            with open(config['predictions'], 'a+') as g:
                g.write("{}\t".format(id))
                for label in labels[j]:
                    g.write("{}\t".format(label))
                g.write("\n")





def main():
    parser = ArgumentParser()
    parser.add_argument("--arch", default='bert', type=str)
    parser.add_argument("--do_data", action='store_true')
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--save_best", action='store_true')
    parser.add_argument("--do_lower_case", action='store_true')
    # parser.add_argument('--data_name', default='HPC', type=str)
    parser.add_argument("--mode", default='min', type=str)
    parser.add_argument("--monitor", default='valid_loss', type=str)

    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--resume_path", default='', type=str)
    parser.add_argument("--predict_checkpoints", type=int, default=0)
    parser.add_argument("--valid_size", default=0.2, type=float)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--sorted", default=1, type=int, help='1 : True  0:False ')
    parser.add_argument("--n_gpu", type=str, default='0', help='"0,1,.." or "0" or "" ')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument('--eval_batch_size', default=8, type=int)
    parser.add_argument("--train_max_seq_len", default=256, type=int)
    parser.add_argument("--eval_max_seq_len", default=256, type=int)
    parser.add_argument('--loss_scale', type=float, default=0)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O1')
    args = parser.parse_args()

    init_logger(log_file=config['log_dir'] / f'{args.arch}-{time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())}.log')
    config['checkpoint_dir'] = config['checkpoint_dir'] / args.arch
    config['checkpoint_dir'].mkdir(exist_ok=True)
    # Good practice: save your training arguments together with the trained model
    torch.save(args, config['checkpoint_dir'] / 'training_args.bin')
    seed_everything(args.seed)
    logger.info("Training/evaluation parameters %s", args)

    if args.do_data:
        data_names = []
        train_sentenses_all = []
        train_target_all = []
        from pybert.io.task_data import TaskData
        data = TaskData()
        total_valid = 0
        for filename in os.listdir(config['summary_path']):
            if filename == ".DS_Store" or filename == "summary":
                continue
            filename_int = int(filename.split('.')[0].split('_')[-1])
            if filename_int > 3500:
                try:
                    raw_data_path = os.path.join(config['summary_path'], filename)
                    # train_targets, train_sentences, val_targets, val_sentences = data.read_data(config,
                    #                                                                             raw_data_path=raw_data_path,
                    #                                                                             preprocessor=EnglishPreProcessor())
                    train_targets, train_sentences, val_targets, val_sentences = data.read_data(config,
                                                                                                raw_data_path=raw_data_path)
                    train_sentenses_all = train_sentenses_all + train_sentences
                    train_target_all = train_target_all + train_targets
                    total_valid = len(train_target_all)
                    print("valid number: ", total_valid)
                    # data.save_pickle(train_sentences, train_targets, data_dir=config['data_dir'],
                    #                  data_name=filename.split('.')[0].split('_')[-1], is_train=True)
                    # data.save_pickle(val_sentences, val_targets, data_dir=config['data_dir'],
                    #                  data_name=filename.split('.')[0].split('_')[-1], is_train=False)

                    # data_names.append(filename.split('.')[0].split('_')[-1])
                except:
                    pass
        total_valid = len(train_target_all)
        print("valid number: ", total_valid)
        data.save_pickle(train_sentenses_all, train_target_all, data_dir=config['data_dir'],
                         data_name="all_valid", is_train=False)

        # with open(config['data_name'], 'w') as f:
        #     json.dump(data_names, f)

    with open(config['data_name'], 'r') as f:
        data_names = json.load(f)

    if args.do_train:
        run_train(args, data_names)

    if args.do_test:
            run_test(args)


    # if args.do_data:
    #     from pybert.io.task_data import TaskData
    #     data = TaskData()
    #     targets, sentences = data.read_data(raw_data_path=config['raw_data_path'],
    #                                         preprocessor=EnglishPreProcessor(),
    #                                         is_train=True)
    #     data.train_val_split(X=sentences, y=targets, shuffle=True, stratify=False,
    #                          valid_size=args.valid_size, data_dir=config['data_dir'],
    #                          data_name=args.data_name)
    # if args.do_train:
    #     run_train(args)
    #
    # if args.do_test:
    #     run_test(args)


if __name__ == '__main__':
    main()
