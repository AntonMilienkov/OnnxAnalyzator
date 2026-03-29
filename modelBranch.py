import torch
import torch.nn as nn

class BranchingModel(nn.Module):
    def __init__(self, d_in=8, d_pre=12, d_hidden=16, d_out=4):
        super().__init__()

        # 1) Слой ДО развилки (общий)
        # Было x размером d_in, стало h размером d_pre
        self.pre = nn.Sequential(
            nn.Linear(d_in, d_pre),
            nn.ReLU(),
        )

        # 2) Две параллельные ветки (обе получают h)
        self.branch1 = nn.Sequential(
            nn.Linear(d_pre, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )
        self.branch2 = nn.Sequential(
            nn.Linear(d_pre, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_out),
        )

        # 3) gate_net тоже получает h и выдаёт число от 0 до 1
        # размер (batch, 1)
        self.gate_net = nn.Sequential(
            nn.Linear(d_pre, 1),
            nn.Sigmoid(),
        )

        # 4) Слой ПОСЛЕ объединения (общий)
        # Он получает y размером d_out и выдаёт финальный y размером d_out
        self.post = nn.Sequential(
            nn.Linear(d_out, d_out),
        )

    def forward(self, x):
        # x: (batch, d_in)

        h = self.pre(x)             # h: (batch, d_pre)

        y1 = self.branch1(h)        # y1: (batch, d_out)
        y2 = self.branch2(h)        # y2: (batch, d_out)

        g = self.gate_net(h)        # g: (batch, 1) числа 0..1

        # объединяем ветки: смесь
        y = g * y1 + (1.0 - g) * y2 # y: (batch, d_out)

        y = self.post(y)            # финальный "после-слой"

        return y, g