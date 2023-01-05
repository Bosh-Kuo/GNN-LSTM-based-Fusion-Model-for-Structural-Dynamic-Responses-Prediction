import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class GCN_Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, gnn_embed_dim, dropout):
        super(GCN_Encoder, self).__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv1 = GCNConv(input_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, hid_dim*2)
        self.conv3 = GCNConv(hid_dim*2, gnn_embed_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = self.dropout(x)
        
        # graph-level embedding
        # gnn_embedding = [batch_size, gnn_embed_dim]
        gnn_embedding = global_mean_pool(x, batch) 
        return gnn_embedding

class LSTM(nn.Module):
    def __init__(self, gnn_embed_dim, hid_dim, n_layers, dropout, pack_mode=True, compression_rate=10, max_story=8):
        super(LSTM, self).__init__()
        assert compression_rate % 10 == 0 and compression_rate >= 10, \
            "compression_rate must be a multiple of 10 and must be bigger than 10!"
        self.gnn_embed_dim = gnn_embed_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.pack_mode = pack_mode
        self.compression_rate = compression_rate  # (必須 ≥ 10, 且為 10 的整數倍，因 src_len = 20000, trg_len = 2000)
        self.output_compression_rate = int(compression_rate / 10)
        self.max_story = max_story
        self.output_dim = max_story * self.output_compression_rate
        self.lstm = nn.LSTM(input_size = gnn_embed_dim + compression_rate,
                            hidden_size=hid_dim, num_layers=n_layers,
                            dropout=dropout, batch_first=True)
        self.fc_out = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, self.output_dim)
        )

    def forward(self, ground_motion, gnn_embedding, seq_length_list = None):
        original_src_seq_len = int(ground_motion.size(-1))
        original_trg_seq_len = int(original_src_seq_len / 10)
        assert self.compression_rate % 10 == 0 and self.compression_rate >= 10 and original_src_seq_len % self.compression_rate == 0, \
            "compression_rate must be a multiple of 10 and must be bigger than 10 and original_src_seq_len is divisible by compression_rate!"
        compressed_seq_len = int(original_src_seq_len / self.compression_rate)
        # ground_motion = [batch_size, src_seq_len/compression_rate, compression_rate]
        ground_motion = torch.reshape(ground_motion, (-1, compressed_seq_len, self.compression_rate))

        # gnn_embedding = [batch_size, src_seq_len/compression_rate, gnn_embed_dim]
        gnn_embedding = gnn_embedding.unsqueeze(1).repeat(1, compressed_seq_len, 1)

        # src = [batch_size, src_seq_len/compression_rate, gnn_embed_dim + compression_rate]
        src = torch.cat((ground_motion, gnn_embedding), dim=2)
        if self.pack_mode:
            compressed_seq_length_list = seq_length_list / self.compression_rate

            packed_src = nn.utils.rnn.pack_padded_sequence(src, lengths = compressed_seq_length_list, batch_first=True, enforce_sorted=False)
            packed_output, (hidden, cell) = self.lstm(packed_src)
            # compress_output = [batch_size, compressed_seq_len, hid_dim]
            compress_output, lens_unpacked = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length = compressed_seq_len)
            # compress_output = [batch_size, compressed_seq_len, max_story * output_compression_rate]
            compress_output = self.fc_out(compress_output)
            # output = [batch, original_trg_seq_len, max_story]
            output = torch.reshape(compress_output, (-1, original_trg_seq_len, self.max_story))
            return output
        else:
            # compress_output = [batch_size, compressed_seq_len, hid_dim]
            compress_output, (hidden, cell) = self.lstm(src)
            # compress_output = [batch_size, compressed_seq_len, max_story * output_compression_rate]
            compress_output = self.fc_out(compress_output)
            # output = [batch, original_trg_seq_len, max_story]
            output = torch.reshape(compress_output, (-1, original_trg_seq_len, self.max_story))
            return output


class GCN_LSTM(nn.Module):
    def __init__(self, GCN_Encoder, LSTM, device):
        super(GCN_LSTM, self).__init__()
        self.GCN_Encoder = GCN_Encoder
        self.LSTM = LSTM
        self.device = device

    def forward(self, Data):
        x = Data.x.to(self.device)
        edge_index = Data.edge_index.to(self.device)
        batch = Data.batch.to(self.device)
        ground_motion = Data.ground_motion.to(self.device)
        seq_length_list = Data.time_steps
        
        # GNN
        # gnn_embedding = [batch_size, gnn_embed_dim]
        gnn_embedding = self.GCN_Encoder(x, edge_index, batch)

        # LSTM
        # output = [batch, original_trg_seq_len, max_story]
        output = self.LSTM(ground_motion, gnn_embedding, seq_length_list)

        return output

class GCN_Only(nn.Module):
    def __init__(self, GCN_Encoder, LSTM, device):
        super(GCN_Only, self).__init__()
        self.GCN_Encoder = GCN_Encoder
        self.LSTM = LSTM
        self.device = device

    def forward(self, Data):
        x = Data.x.to(self.device)
        edge_index = Data.edge_index.to(self.device)
        batch = Data.batch.to(self.device)
        ground_motion = Data.ground_motion.to(self.device)
        seq_length_list = Data.time_steps
        
        # GNN
        # gnn_embedding = [batch_size, gnn_embed_dim]
        gnn_embedding = self.GCN_Encoder(x, edge_index, batch)

        # LSTM
        # output = [batch, original_trg_seq_len, max_story]
        # output = self.LSTM(ground_motion, gnn_embedding, seq_length_list)

        return gnn_embedding
