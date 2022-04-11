import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class trainer:
    def __init__(self, encoder, mlp, lstm, reward_heads, train_loader, validate_loader, epoch_size=10, learning_rate=1e-2, max_len=50):
        self.encoder = encoder
        self.mlp = mlp
        self.lstm = lstm
        self.reward_heads = reward_heads
        self.max_len = max_len
        self.model = Network(encoder, mlp, lstm, reward_heads, self.max_len).to(device)

        self.train_loader = train_loader
        self.validate_loader = validate_loader
        self.epoch_size = epoch_size 
        self.learning_rate = learning_rate
        self.validation_epoch = 1  # validation 빈도
        self.criterion = nn.MSELoss()
        self.train_writer = SummaryWriter('logs/reward_prediction/train/')
        self.val_writer = SummaryWriter('logs/reward_prediction/validate/')
        self.val_t = 0

    def train(self):
        optimizer = optim.RAdam(self.model.parameters(), lr=self.learning_rate)

        min_epoch_loss = 1e10
        t = 0
        for epoch in range(self.epoch_size):
            self.model.train()
            
            epoch_loss = 0
            
            for idx, sample in enumerate(self.train_loader):
                t += 1
                inputs = sample['images'][0]
                target_rewards = sample['rewards']
 
                optimizer.zero_grad()

                outputs = self.model(inputs) * 10

                target_r = []
                for key in target_rewards.keys():
                    target_r.append(target_rewards[key])
                targets = torch.stack(target_r, axis = 2).to(device) * 10

                loss = self.criterion(outputs, targets)

                loss.backward()

                optimizer.step()

                epoch_loss += loss
                self.train_writer.add_scalar("loss", loss.item(), t) # recording
                
                if idx % 1000 == 0:
                    print('Estimated : ', outputs.view(-1, self.max_len))
                    print('Target : ', targets.view(-1, self.max_len))
                    print(f"{idx + 1} step loss : {loss.item()}")

            print(f"{epoch + 1} eopch loss : {epoch_loss.item() / len(self.train_loader)}")

            if min_epoch_loss > epoch_loss / len(self.train_loader):
                min_epoch_loss = epoch_loss / len(self.train_loader)
                torch.save(self.encoder.state_dict(), './model/Reward_prediction_encoder.pt')
                print('New model saved!')
            
            if (epoch + 1) % self.validation_epoch == 0:
                validation_loss = self.validate(self.validate_loader)
                print(f"Validation loss : {validation_loss}")

        self.train_writer.flush()
        self.val_writer.flush()

        self.train_writer.close()
        self.val_writer.close()

    @torch.no_grad()
    def validate(self, validate_loader):
        self.model.eval()
        total_loss = 0
        for idx, sample in enumerate(validate_loader):
            self.val_t += 1
            inputs = sample['images'][0]
            target_rewards = sample['rewards']

            outputs = self.model(inputs) * 10

            target_r = []
            for key in target_rewards.keys():
                target_r.append(target_rewards[key])
            targets = torch.stack(target_r, axis = 2).to(device) * 10

            loss = self.criterion(outputs, targets)

            self.val_writer.add_scalar("loss", loss, self.val_t)

            total_loss += loss
            
        return total_loss.item() / len(validate_loader)
    

class Encoder(nn.Module):
    def __init__(self, input_size=1, output_size=64):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_size, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(128, output_size)

    def forward(self, x):
        x = x.to(device)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, max_len):
        super(Decoder, self).__init__()
        self.max_len = max_len
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
 
        self.fc = nn.Linear(64, 128)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.fc(x))
        x = x.view(self.max_len, 128, 1, 1)
        x = self.deconv(x)          
        return x


class MLP(nn.Module):
    def __init__(self, input_size=64, output_size=64):
        super(MLP, self).__init__()
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 입력 : (batch, timestep, dimension) / 출력 : (output, (h, c))
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=64, hidden_size=64)

    def forward(self, x):
        x = x.to(device)
        x, _ = self.lstm(x)
        return x


class Network(nn.Module):
    def __init__(self, encoder, mlp, lstm, reward_heads, max_len):
        super().__init__()
        self.encoder = encoder
        self.mlp = mlp
        self.lstm = lstm
        self.reward_heads = reward_heads
        self.max_len = max_len

    def forward(self, x):
        x = x.to(device)
        outputs = self.encoder(x)
        outputs = self.mlp(outputs)
        outputs = outputs.view(1, self.max_len, -1)
        outputs = self.lstm(outputs)
        outputs = self.reward_heads(outputs)
        return outputs