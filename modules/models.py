import random
import torch
from torch import nn
import torch.nn.functional as F
import timm
from torchvision import transforms as TT
from torchlibrosa.augmentation import SpecAugmentation
from nnAudio.Spectrogram import CQT1992v2, CQT2010v2

class CustomModel(nn.Module):
    def __init__(self, model_name, in_chans=3, reshape_factor=128, drop_rate=0.1, drop_path_rate=0.1,
                 num_classes: int = 6, pretrained: bool = True):
        super(CustomModel, self).__init__()
        self.backbone = timm.create_model(
            model_name,
            in_chans=in_chans,
            pretrained=pretrained,
            drop_rate = drop_rate,
            drop_path_rate = drop_path_rate,
        )
        self.reshape_factor = reshape_factor

        if 'efficient' in model_name:
            self.backbone.global_pool = nn.Identity()
            self.backbone.classifier = nn.Identity()
            self.fc = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.backbone.num_features, num_classes)
            )

        elif "convnext" in model_name:
            self.backbone.head.fc = nn.Identity()
            self.fc = nn.Linear(self.backbone.num_features, num_classes)

        elif "nfnet" in model_name:
            self.backbone.head = nn.Identity()
            self.fc = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(drop_rate),
                nn.Linear(self.backbone.num_features, num_classes)
            )
        elif model_name == "tiny_vit_21m_512":
            self.backbone.head.fc = nn.Identity()
            self.fc = nn.Linear(self.backbone.num_features, num_classes)

        elif "maxvit_" in model_name:
            self.backbone.head.fc = nn.Identity()
            self.fc = nn.Linear(self.backbone.num_features, num_classes)

        elif "vit_" in model_name:
            self.backbone.head = nn.Identity()
            self.fc = nn.Linear(self.backbone.num_features, num_classes)

    def __reshape(self, x):
        bs, d = x.shape
        reshaped_tensor = x.view(bs, d//self.reshape_factor, self.reshape_factor)
        x = torch.unsqueeze(reshaped_tensor, dim=1)
        x = torch.cat([x, x, x], dim=1)
        return x

    def forward(self, x):
        x = self.__reshape(x)
        x = self.backbone(x) # (bs, ch, feat_size, feat_size)
        x = self.fc(x) # (bs, 6)
        return x
    
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'

class Net(nn.Module):
    def __init__(self, base_model_name: str, pretrained=False, num_classes=24, in_channels=1):
        super().__init__()
        self.qtransform = CQT1992v2(sr=32000, fmin=256, n_bins=160, hop_length=250, output_format='Magnitude',
                                    norm=1, window='tukey',bins_per_octave=27, verbose=False)
        self.spec_augmenter = SpecAugmentation(time_drop_width=64//2, time_stripes_num=2,
                                               freq_drop_width=8//2, freq_stripes_num=2)

        base_model = timm.create_model(
            base_model_name, pretrained=pretrained, in_chans=in_channels)
        self.encoder = base_model#nn.Sequential(*layers)
        if hasattr(base_model, "fc"):
            in_features = base_model.fc.in_features
        elif hasattr(base_model, "classifier"):
            in_features = base_model.classifier.in_features
            self.encoder.global_pool = nn.Identity()
            self.encoder.classifier = nn.Identity()
        else:
            in_features = base_model.head.fc.in_features
        self.pooling = GeM()
        self.fc = nn.Linear(in_features, num_classes)
        
    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)


    def forward(self, input_data):
        x = input_data # (batch_size, 3, time_steps, mel_bins)
        x = self.qtransform(x)
        x = x.unsqueeze(1)

        batch_size = x.size(0)
        
        if self.training:
            if random.random() < 0.25:
                x = self.spec_augmenter(x)

        x = self.encoder(x)
        x = self.pooling(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)

        return x

def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)


def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orghogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()


def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    output = F.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear").squeeze(1)

    return output


class AttBlockV2(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)


class TimmSED(nn.Module):
    def __init__(self, base_model_name: str, pretrained=False, num_classes=24, in_channels=1, img_size=(320, 641)):
        super().__init__()

        self.qtransform = CQT1992v2(sr=32000, fmin=256, n_bins=160, hop_length=250, output_format='Magnitude',
                                    norm=1, window='tukey',bins_per_octave=27, verbose=False)

        self.spec_augmenter = SpecAugmentation(time_drop_width=64//2, time_stripes_num=2,
                                               freq_drop_width=8//2, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(320) # 641 CFG.n_mels
        self.resize = TT.Resize([img_size[0], img_size[1]])
        base_model = timm.create_model(
            base_model_name, pretrained=pretrained, in_chans=in_channels)
        self.encoder = base_model#nn.Sequential(*layers)
        if hasattr(base_model, "fc"):
            in_features = base_model.fc.in_features
        elif hasattr(base_model, "classifier"):
            in_features = base_model.classifier.in_features
            self.encoder.global_pool = nn.Identity()
            self.encoder.classifier = nn.Identity()
        else:
            in_features = base_model.head.fc.in_features

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(
            in_features, num_classes, activation="sigmoid")

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)


    def forward(self, input_data):
        x = input_data # (batch_size, 3, time_steps, mel_bins)
        x = self.qtransform(x)
        x = (64*x + 1).log()
        x = self.resize(x)
        x = x.unsqueeze(1)
        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            if random.random() < 0.25:
                x = self.spec_augmenter(x)

        x = x.transpose(2, 3)

        x = self.encoder(x)
        # Aggregate in frequency axis
        x = torch.mean(x, dim=3)

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        segmentwise_logit = self.att_block.cla(x).transpose(1, 2)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        interpolate_ratio = frames_num // segmentwise_output.size(1)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output,
                                       interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)

        output_dict = {
            'framewise_output': framewise_output,
            'clipwise_output': clipwise_output,
            'logit': logit,
            'framewise_logit': framewise_logit,
        }

        return output_dict