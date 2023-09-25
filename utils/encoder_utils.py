from torch import nn


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Res64(nn.Module):
    def __init__(self, k, p, l=64, conv2d=False, conv2d_mix=False):
        super().__init__()
        self.k = k
        self.p = p

        if conv2d:
            self.layer = nn.Sequential(
                nn.Conv2d(l, l, stride=(1, 1), kernel_size=(
                    1, self.k), padding=(0, self.p)),
                nn.BatchNorm2d(l),
                nn.LeakyReLU(0.01))
        elif conv2d_mix:
            self.layer = nn.Sequential(
                nn.Conv2d(l, l, stride=(1, 1), kernel_size=(
                    3, self.k), padding=(1, self.p)),
                nn.BatchNorm2d(l),
                nn.LeakyReLU(0.01))

        else:
            self.layer = nn.Sequential(
                nn.Conv1d(l, l, stride=1, kernel_size=self.k, padding=self.p),
                nn.BatchNorm1d(l),
                nn.LeakyReLU(0.01))

    def forward(self, x):
        identity = x

        out = self.layer(x)
        out = self.layer(out)
        out += identity

        return out


class Res64Transpose(nn.Module):
    def __init__(self, k, p, l=64, conv2d=False, conv2d_mix=False):
        super().__init__()
        self.k = k
        self.p = p

        if conv2d:
            self.layer = nn.Sequential(
                nn.ConvTranspose2d(l, l, stride=(1, 1), kernel_size=(
                    1, self.k), padding=(0, self.p)),
                nn.LeakyReLU(0.01))
        elif conv2d_mix:
            self.layer = nn.Sequential(
                nn.ConvTranspose2d(l, l, stride=(1, 1), kernel_size=(
                    3, self.k), padding=(1, self.p)),
                nn.LeakyReLU(0.01))
        else:
            self.layer = nn.Sequential(
                nn.ConvTranspose1d(
                    l, l, stride=1, kernel_size=self.k, padding=self.p),
                nn.LeakyReLU(0.01))

    def forward(self, x):
        identity = x

        out = self.layer(x)
        out = self.layer(out)
        out += identity

        return out


class Layer64(nn.Module):
    def __init__(self, k, p, n, conv2d=False, conv1d_12=False):
        super().__init__()
        self.k = k
        self.p = p
        self.n = n
        self.conv2d = conv2d
        self.conv1d_12 = conv1d_12

        if self.conv2d:
            self.layer = nn.Sequential(
                nn.Conv2d(64, 64, stride=(1, 2), kernel_size=(
                    1, self.k), padding=(0, self.p)),
                nn.LeakyReLU(0.01))
        elif self.conv1d_12:
            self.layer = nn.Sequential(
                nn.Conv1d(64*12, 64*12, stride=(2),
                          kernel_size=self.k, padding=self.p, groups=12),
                nn.LeakyReLU(0.01))
        else:
            self.layer = nn.Sequential(
                nn.Conv1d(64, 64, stride=(2),
                          kernel_size=self.k, padding=self.p),
                nn.LeakyReLU(0.01))

    def forward(self, x):
        for _ in range(self.n):
            x = self.layer(x)
        return x


class Layer64Transpose(nn.Module):
    def __init__(self, k, p, n, conv2d=False, conv1d_12=False):
        super().__init__()
        self.k = k
        self.p = p
        self.n = n
        self.conv2d = conv2d
        self.conv1d_12 = conv1d_12

        if self.conv2d:
            self.layer = nn.Sequential(
                nn.ConvTranspose2d(64, 64, stride=(1, 2), kernel_size=(
                    1, self.k), padding=(0, self.p)),
                nn.LeakyReLU(0.01))

        elif self.conv1d_12:
            self.layer = nn.Sequential(
                nn.ConvTranspose1d(64*12, 64*12, stride=(2),
                                   kernel_size=self.k, padding=self.p, groups=12),
                nn.LeakyReLU(0.01))
        else:
            self.layer = nn.Sequential(
                nn.ConvTranspose1d(64, 64, stride=(
                    2), kernel_size=self.k, padding=self.p),
                nn.LeakyReLU(0.01))

    def forward(self, x):
        for _ in range(self.n):
            x = self.layer(x)
        return x
