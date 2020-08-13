
from pathlib import Path
BASE_DIR = Path('/home/home1/xw176/work/Bert-Multi-Label-Text-Classification/pybert')
config = {
    # 'raw_data_path': BASE_DIR / 'dataset/train.csv',
    'test_path': BASE_DIR / 'dataset/data/summary/Test.csv',
    # 'test_path': BASE_DIR / 'dataset/summary/summary_1128.csv',
    'summary_path': BASE_DIR / 'dataset/summary',
    'subclass_list': BASE_DIR / 'dataset/subclass.json',
    'data_name': BASE_DIR / 'dataset/data_name.json',

    'data_dir': BASE_DIR / 'dataset/summary_pickle',
    'cached_dir': BASE_DIR / 'dataset/cached',
    'log_dir': BASE_DIR / 'output/log',
    'writer_dir': BASE_DIR / "output/TSboard",
    'figure_dir': BASE_DIR / "output/figure",
    'checkpoint_dir': BASE_DIR / "output/checkpoints",
    'cache_dir': BASE_DIR / 'model/',
    'result': BASE_DIR / "output/result",
    'predictions': BASE_DIR / "output/predictions.txt"

    # 'bert_vocab_path': BASE_DIR / 'pretrain/bert/base-uncased/bert_vocab.txt',
    # 'bert_config_file': BASE_DIR / 'pretrain/bert/base-uncased/config.json',
    # 'bert_model_dir': BASE_DIR / 'pretrain/bert/base-uncased',

    # 'xlnet_vocab_path': BASE_DIR / 'pretrain/xlnet/base-cased/spiece.model',
    # 'xlnet_config_file': BASE_DIR / 'pretrain/xlnet/base-cased/config.json',
    # 'xlnet_model_dir': BASE_DIR / 'pretrain/xlnet/base-cased',
    #
    # 'albert_vocab_path': BASE_DIR / 'pretrain/albert/albert-base/30k-clean.model',
    # 'albert_config_file': BASE_DIR / 'pretrain/albert/albert-base/config.json',
    # 'albert_model_dir': BASE_DIR / 'pretrain/albert/albert-base'


}

