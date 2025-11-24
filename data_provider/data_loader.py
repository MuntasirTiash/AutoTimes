import os
import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from data_provider.m4 import M4Dataset, M4Meta
from sklearn.preprocessing import StandardScaler
from utils.tools import convert_tsf_to_dataframe
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv',
                 scale=True, seasonal_patterns=None, drop_short=False):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        data_name = self.data_path.split('.')[0]
        self.data_stamp = torch.load(os.path.join(self.root_path, f'{data_name}.pt'))
        self.data_stamp = self.data_stamp[border1:border2]
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.time_index = pd.to_datetime(df_raw.iloc[border1:border2, 0]).to_numpy()

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = self.data_stamp[s_begin:s_end:self.token_len]
        seq_y_mark = self.data_stamp[s_end:r_end:self.token_len]

        is_test = (getattr(self, 'set_type', None) == 'test') or (getattr(self, 'flag', None) == 'test')
        if is_test:
            # label window timestamps, aligned to seq_y
            seq_y_time = self.time_index[r_begin:r_end]
            # convert to int64 nanoseconds so DataLoader can collate as LongTensor
            seq_y_time = seq_y_time.astype('datetime64[ns]').astype('int64')
            return seq_x, seq_y, seq_x_mark, seq_y_mark, torch.from_numpy(seq_y_time)

        # train/val unchanged
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv',
                 scale=True, seasonal_patterns=None, drop_short=False):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
            

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        data_name = self.data_path.split('.')[0]
        self.data_stamp = torch.load(os.path.join(self.root_path, f'{data_name}.pt'))
        self.data_stamp = self.data_stamp[border1:border2]
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = self.data_stamp[s_begin:s_end:self.token_len]
        seq_y_mark = self.data_stamp[s_end:r_end:self.token_len]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv',
                 seasonal_patterns=None, scale=True, drop_short=False):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, root_path, flag='pred', size=None, data_path='ETTh1.csv',
                 scale=False, inverse=False, seasonal_patterns='Yearly', drop_short=False):
        self.scale = scale
        self.inverse = inverse
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    # def __read_data__(self):
    #     # M4Dataset.initialize()
    #     if self.flag == 'train':
    #         dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
    #     else:
    #         dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
    #     # training_values = np.array(
    #     #     [v[~np.isnan(v)] for v in
    #     #      dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
    #     # self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
    #     # self.timeseries = [ts for ts in training_values]
    #     training_values = [np.asarray(v[~np.isnan(v)], dtype=np.float32)
    #                     for v in dataset.values[dataset.groups == self.seasonal_patterns]]
        
    #     self.timeseries = training_values

    def __read_data__(self):
        # Load the split
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)

        # Keep only the chosen seasonal pattern
        mask = (dataset.groups == self.seasonal_patterns)

        # ---- IDs (must align with timeseries order) ----
        ids = dataset.ids[mask]
        # normalize dtype to string/object (handles possible bytes)
        try:
            ids = ids.astype(str)
        except Exception:
            ids = np.array([str(i) for i in ids], dtype=object)
        self.ids = np.asarray(ids, dtype=object)  # e.g., ['Y1','Y2',...]

        # ---- Values as a ragged list of 1-D float arrays ----
        series_list = []
        ids_list = []
        for i, v in zip(self.ids, dataset.values[mask]):
            arr = np.asarray(v[~np.isnan(v)], dtype=np.float32)
            if arr.size == 0:
                # drop empty series to keep alignment with predictions
                continue
            series_list.append(arr)
            ids_list.append(i)

        self.timeseries = series_list
        self.ids = np.asarray(ids_list, dtype=object)

        # Optional sanity check:
        assert len(self.timeseries) == len(self.ids), "ids and timeseries length mismatch"

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class Dataset_TSF(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path=None,
                 scale=True, seasonal_patterns=None, drop_short=False):
        
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.pred_len
        self.context_len = 4 * self.token_len
        print(self.seq_len, self.label_len, self.pred_len)
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.data_path = data_path
        self.drop_short = drop_short
        self.timeseries = self.__read_data__()


    def __read_data__(self):
        df, _, _, _, _ = convert_tsf_to_dataframe(os.path.join(self.root_path, self.data_path))
        def dropna(x):
            return x[~np.isnan(x)]
        timeseries = [dropna(ts).astype(np.float32) for ts in df.series_value]
        if self.drop_short:
            timeseries = [ts for ts in timeseries if ts.shape[0] > self.context_len]
        self.tot_len = 0
        self.len_seq = []
        self.seq_id = []
        for i in range(len(timeseries)):
            res_len = max(self.pred_len + self.seq_len - timeseries[i].shape[0], 0)
            pad_zeros = np.zeros(res_len)
            timeseries[i] = np.hstack([pad_zeros, timeseries[i]])

            _len = timeseries[i].shape[0]
            train_len = _len-self.pred_len
            border1s = [0,                          train_len - self.seq_len - self.pred_len, train_len-self.seq_len]
            border2s = [train_len - self.pred_len,  train_len,                                _len]
            
            curr_len = border2s[self.set_type] - max(border1s[self.set_type], 0) - self.pred_len - self.seq_len + 1
            curr_len = max(0, curr_len)
            
            self.len_seq.append(np.zeros(curr_len) + self.tot_len)
            self.seq_id.append(np.zeros(curr_len) + i)
            self.tot_len += curr_len
            
        self.len_seq = np.hstack(self.len_seq)
        self.seq_id = np.hstack(self.seq_id)

        return timeseries

    def __getitem__(self, index):
        len_seq = self.len_seq[index]
        seq_id = int(self.seq_id[index])
        index = index - int(len_seq)

        _len = self.timeseries[seq_id].shape[0]
        train_len = _len - self.pred_len
        border1s = [0,                          train_len - self.seq_len - self.pred_len, train_len-self.seq_len]

        s_begin = index + border1s[self.set_type]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        data_x = self.timeseries[seq_id][s_begin:s_end]
        data_y = self.timeseries[seq_id][r_begin:r_end]
        data_x = np.expand_dims(data_x, axis=-1)
        data_y = np.expand_dims(data_y, axis=-1)

        return data_x, data_y, data_x, data_y

    def __len__(self):
        return self.tot_len

class Dataset_TSF_ICL(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path=None,
                 scale=True, seasonal_patterns=None, drop_short=True):
        
        self.pred_len = size[2]
        self.token_len = self.pred_len
        self.context_len = 4 * self.token_len

        self.root_path = root_path
        self.data_path = data_path
        self.timeseries = self.__read_data__()

    def __read_data__(self):
        df, _, _, _, _ = convert_tsf_to_dataframe(os.path.join(self.root_path, self.data_path))
        def dropna(x):
            return x[~np.isnan(x)]
        timeseries = [dropna(ts).astype(np.float32) for ts in df.series_value]
        timeseries = [ts for ts in timeseries if ts.shape[0] > self.context_len]
        return timeseries

    # we uniformly adopting the first time points of the time series as the corresponding prompt.
    def __getitem__(self, index):        
        data_x1 = self.timeseries[index][:2*self.token_len]
        data_x2 = self.timeseries[index][-2*self.token_len:-1*self.token_len]
        data_x = np.concatenate((data_x1, data_x2))
        data_y = self.timeseries[index][-1*self.token_len:]
        data_x = np.expand_dims(data_x, axis=-1)
        data_y = np.expand_dims(data_y, axis=-1)
        return data_x, data_y, data_x, data_y

    def __len__(self):
        return len(self.timeseries)

class Dataset_Preprocess(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='ETTh1.csv', scale=True, seasonal_patterns=None):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        self.data_set_type = data_path.split('.')[0]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.tot_len = len(self.data_stamp)

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date).apply(str)
        self.data_stamp = df_stamp['date'].values
        self.data_stamp = [str(x) for x in self.data_stamp]
        

    def __getitem__(self, index):
        s_begin = index % self.tot_len
        s_end = s_begin + self.token_len
        start = datetime.datetime.strptime(self.data_stamp[s_begin], "%Y-%m-%d %H:%M:%S")
        if self.data_set_type in ['traffic', 'electricity', 'ETTh1', 'ETTh2']:
            end = (start + datetime.timedelta(hours=self.token_len-1)).strftime("%Y-%m-%d %H:%M:%S")
        elif self.data_set_type == 'weather':
            end = (start + datetime.timedelta(minutes=10*(self.token_len-1))).strftime("%Y-%m-%d %H:%M:%S")
        elif self.data_set_type in ['ETTm1', 'ETTm2']:
            end = (start + datetime.timedelta(minutes=15*(self.token_len-1))).strftime("%Y-%m-%d %H:%M:%S")
        seq_x_mark = f"This is Time Series from {self.data_stamp[s_begin]} to {end}"
        return seq_x_mark

    def __len__(self):
        return len(self.data_stamp)

    
class Dataset_EPSPanel(Dataset):
    """
    Panel EPS dataset:
      - CSV with columns: gvkey, DATE, actual
      - Univariate EPS forecasting (no extra features)
      - Window-based train/val/test split:
          * history can use all past data
          * split is decided by the date of the FIRST predicted quarter
      - Z-scales EPS using TRAIN subset only, with inverse_transform support.
    """
    def __init__(self, root_path, flag='train', size=None,
                 data_path='eps_panel.csv',
                 scale=True,
                 firm_col='gvkey',
                 date_col='DATE',
                 target_col='actual',
                 train_end='2014-12-31',
                 val_end='2017-12-31',
                 return_meta=False):
        """
        size: (seq_len, label_len, pred_len)
        flag: 'train', 'val', or 'test'
        train_end, val_end: split boundaries (based on FIRST predicted quarter)
        return_meta:
          - False: __getitem__ returns (seq_x, seq_y, seq_x_mark, seq_y_mark)
          - True:  returns (..., firm_id, target_quarters)
        """
        assert flag in ['train', 'val', 'test']
        self.flag = flag
        self.return_meta = return_meta

        if size is None:
            # you will usually override this via args
            self.seq_len = 16
            self.label_len = 8
            self.pred_len = 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path

        self.firm_col = firm_col
        self.date_col = date_col
        self.target_col = target_col

        self.train_end = pd.to_datetime(train_end)
        self.val_end = pd.to_datetime(val_end)

        # scaler + stats
        self.scaler = None
        self.eps_mean = None
        self.eps_std = None

        # per-firm storage
        self.series_values = []    # original EPS per firm (np.array)
        self.series_scaled = []    # scaled EPS per firm (np.array)
        self.series_dates = []     # np.array of datetimes per firm
        self.series_firm_ids = []  # firm_id per firm

        # window index list: each element is (series_idx, start_pos)
        self.indices = []

        self.__read_data__()

        # univariate series
        self.enc_in = 1

    def __read_data__(self):
        # ----- load full panel for ALL splits -----
        df = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df = df[[self.firm_col, self.date_col, self.target_col]].dropna()

        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df = df.sort_values([self.firm_col, self.date_col])

        # ----- fit scaler on TRAIN subset only -----
        if self.scale:
            from sklearn.preprocessing import StandardScaler
            train_mask = df[self.date_col] <= self.train_end
            df_train = df.loc[train_mask]

            self.scaler = StandardScaler()
            self.scaler.fit(df_train[[self.target_col]].values)

            # cache mean/std for convenience
            self.eps_mean = float(self.scaler.mean_[0])
            self.eps_std = float(self.scaler.scale_[0])
        else:
            self.scaler = None
            self.eps_mean = 0.0
            self.eps_std = 1.0

        # ----- build full per-firm series (values + dates) -----
        for firm_id, g in df.groupby(self.firm_col):
            vals_all = g[self.target_col].values.astype('float32')
            dates_all = g[self.date_col].values

            # need at least one full window somewhere
            if len(vals_all) < self.seq_len + self.pred_len:
                continue

            if self.scale:
                vals_scaled = self.scaler.transform(
                    vals_all.reshape(-1, 1)
                ).reshape(-1).astype('float32')
            else:
                vals_scaled = vals_all.copy()

            self.series_values.append(vals_all)      # original EPS
            self.series_scaled.append(vals_scaled)   # scaled EPS
            self.series_dates.append(dates_all)
            self.series_firm_ids.append(firm_id)

        # ----- build window indices and assign to train/val/test -----
        self.indices = []
        for s_idx, vals_scaled in enumerate(self.series_scaled):
            dates_all = self.series_dates[s_idx]
            L = len(vals_scaled)

            # possible start positions for history window
            max_start = L - (self.seq_len + self.pred_len) + 1
            if max_start < 1:
                continue

            for start in range(max_start):
                # define the standard AutoTimes segmentation
                s_begin = start
                s_end = start + self.seq_len

                r_begin = s_end - self.label_len
                r_end = r_begin + self.label_len + self.pred_len
                # safety check
                if r_end > L:
                    continue

                # forecast part is the last pred_len of [r_begin:r_end]
                pred_start_idx = r_begin + self.label_len
                d_first_pred = dates_all[pred_start_idx]

                # assign window based on FIRST predicted quarter
                if self.flag == 'train':
                    if d_first_pred <= self.train_end:
                        self.indices.append((s_idx, start))
                elif self.flag == 'val':
                    if (d_first_pred > self.train_end) and (d_first_pred <= self.val_end):
                        self.indices.append((s_idx, start))
                else:  # 'test'
                    if d_first_pred > self.val_end:
                        self.indices.append((s_idx, start))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        s_idx, start = self.indices[index]

        vals_scaled = self.series_scaled[s_idx]
        dates = self.series_dates[s_idx]
        firm_id = self.series_firm_ids[s_idx]

        s_begin = start
        s_end = start + self.seq_len

        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # sequences in z-scale (univariate)
        seq_x = vals_scaled[s_begin:s_end].reshape(-1, 1)          # [seq_len, 1]
        seq_y = vals_scaled[r_begin:r_end].reshape(-1, 1)          # [label_len+pred_len, 1]

        # corresponding dates (label + pred window)
        target_quarters = dates[r_begin:r_end]

        # no time features yet (can add later)
        seq_x_mark = np.zeros((self.seq_len, 1), dtype='float32')
        seq_y_mark = np.zeros((self.label_len + self.pred_len, 1), dtype='float32')

        if self.return_meta:
            # used in test: we want to know firm_id and exact quarters
            return seq_x, seq_y, seq_x_mark, seq_y_mark, firm_id, target_quarters
        else:
            # used in train/val: keep exactly the 4-tuple interface
            return seq_x, seq_y, seq_x_mark, seq_y_mark

    def inverse_transform(self, data):
        """
        data: array or tensor of EPS in z-scale.
        returns: same shape in original EPS units.
        """
        if not self.scale:
            return data
        return data * self.eps_std + self.eps_mean