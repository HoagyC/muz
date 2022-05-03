import torch
from torch import nn


class LSTMTest(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, batch_first=False
        )

    def forward(self, x, h0, c0):
        out = self.lstm(x, (h0, c0))
        return out


class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, cell_init: torch.Tensor = None):
        super().__init__()
        # Note: input size is number of features, not batch

        self.in_i = nn.Linear(input_size, hidden_size)
        self.in_h = nn.Linear(hidden_size, hidden_size)

        self.f_i = nn.Linear(input_size, hidden_size)
        self.f_h = nn.Linear(hidden_size, hidden_size)

        self.g_i = nn.Linear(input_size, hidden_size)
        self.g_h = nn.Linear(hidden_size, hidden_size)

        self.o_i = nn.Linear(input_size, hidden_size)
        self.o_h = nn.Linear(hidden_size, hidden_size)

        if cell_init:
            assert cell_init.shape == hidden_size
            self.cell = cell_init
        else:
            self.cell = torch.zeros(hidden_size)

    def forward(self, x, hidden):
        input_gate = torch.softmax(self.in_i(x) + self.in_h(hidden), dim=1)
        forget_gate = torch.softmax(self.f_i(x) + self.f_h(hidden), dim=1)

        cell_gate = torch.tanh(self.g_i(x) + self.g_h(hidden))
        output_gate = torch.softmax(self.o_i(x) + self.o_h(hidden), dim=1)

        self.cell = forget_gate * self.cell + input_gate * cell_gate
        self.hidden = output_gate * torch.tanh(cell_gate)
        return self.hidden


class LSTMbyHand(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.num_layers = num_layers

        assert hidden_size > 0

        self.lstm_layers = [
            LSTMLayer(self.input_size if i == 0 else self.hidden_size, self.hidden_size)
            for i in range(self.num_layers)
        ]

    def forward(self, input_t: torch.Tensor, init_hiddens):
        # Check shape of input tensor
        no_batch = False
        if input_t.ndim == 2:
            no_batch = True
            input_t.unsqueeze_(1)
        elif self.batch_first:
            input_t = input_t.permute(1, 0, 2)

        seq_len, batch_size, input_size = input_t.shape
        assert input_size == self.input_size

        # Check shape of hidden tensors
        if init_hiddens:
            h0, c0 = init_hiddens
        else:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        if c0.ndim == 2:
            assert h0.ndim == 2
            assert batch_size == 1
            assert no_batch
            c0.unsqueeze_(1)
            h0.unsqueeze_(1)

        assert h0.shape == (self.num_layers, batch_size, self.hidden_size)
        assert c0.shape == (self.num_layers, batch_size, self.hidden_size)

        # Put in initial hidden states
        for i, layer in enumerate(self.lstm_layers):
            layer.cell = c0[i]

        hidden_t = h0
        output_t = torch.zeros(seq_len, batch_size, hidden_size)
        for i in range(seq_len):
            for j, layer in enumerate(self.lstm_layers):
                if j == 0:
                    hidden_t[j] = layer(input_t[i], hidden_t[0])
                else:
                    hidden_t[j] = layer(hidden_t[j - 1], hidden_t[j])

                if j == self.num_layers - 1:
                    output_t[i] = hidden_t[j]

        hn = torch.stack([layer.hidden for layer in self.lstm_layers])
        cn = torch.stack([layer.cell for layer in self.lstm_layers])

        assert output_t.shape == (seq_len, batch_size, self.hidden_size)
        assert hn.shape == (self.num_layers, batch_size, self.hidden_size)
        assert cn.shape == (self.num_layers, batch_size, self.hidden_size)

        if no_batch:
            output_t.squeeze_(1)
            hn.squeeze_(1)
            cn.squeeze_(1)

        return output_t, (hn, cn)


if __name__ == "__main__":

    input_size = 10
    hidden_size = 20

    num_layers = 2
    batch_size = 3
    seq_len = 50

    rnn = nn.LSTM(input_size, hidden_size, num_layers)
    home_rnn = LSTMbyHand(input_size, hidden_size, num_layers)

    input_t = torch.ones((seq_len, batch_size, input_size))
    h0 = torch.ones((num_layers, batch_size, hidden_size))
    c0 = torch.ones((num_layers, batch_size, hidden_size))
    output, (hn, cn) = rnn(input_t, (h0, c0))
    home_output, (hhn, hcn) = home_rnn(input_t, (h0, c0))
    print(home_output, output)
    print(home_output.var(), output.var())
