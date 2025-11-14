# debug_ett_times.py
from types import SimpleNamespace as NS
from data_provider.data_factory import data_provider

# Mirror the flags you use in your ETTh1 runs (+ a few commonly required ones)
args = NS(
    # dataset/task identity
    task_name='long_term_forecast',
    is_training=0,
    data='ETTh1',

    # paths
    root_path='./dataset/ETT-small/',
    data_path='ETTh1.csv',

    # window sizes
    seq_len=672, label_len=576, pred_len=96,
    test_seq_len=672, test_label_len=576, test_pred_len=96,

    # batching / loader
    batch_size=8, num_workers=0, drop_last=True, shuffle=False,

    # time encoding / misc parms used by these repos
    token_len=96,
    features='M',        # multivariate
    target='OT',         # standard ETTh target label name in many repos
    freq='h',            # hourly
    seasonal_patterns='none',  # just needs to exist; value not used by ETTh1 path
    mix_embeds=True,     # since you use --mix_embeds in runs
    drop_short = False,

    # placeholders that some codepaths may read (harmless defaults)
    use_amp=False, inverse=False, scale=True, use_multi_gpu= False
)

test_data, test_loader = data_provider(args, 'test')
print("dataset:", type(test_data).__name__, 
      "| flag:", getattr(test_data, 'set_type', getattr(test_data, 'flag', None)))
print("has time_index:", hasattr(test_data, 'time_index'))

batch = next(iter(test_loader))
print("len(batch) =", len(batch))
for i, x in enumerate(batch):
    shape = getattr(x, "shape", None)
    dtype = getattr(x, "dtype", None)
    print(f"  item {i}: shape={shape}, dtype={dtype}, type={type(x)}")