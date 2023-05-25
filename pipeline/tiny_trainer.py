import torch
import torch.nn as nn
import torch.optim as optim


# from ..utils.progress_bar import progress_bar
from tqdm.auto import tqdm
# from tqdm import tqdm_notebook as tqdm


class TinyTrainer():
    def __init__(self, model, device=None):
        self.model = model
        
        self.device = device
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        

    def before_loss(self, loss, inputs, labels):
        pass

    def after_inference(self):
        pass

    def train(self, train_loader, val_loader, num_epochs, lr, lr_scheduler=None, weight_decay=5e-4, criterion=None):
        device = self.device
        self.model = self.model.to(device)

        if not criterion:
            criterion = nn.CrossEntropyLoss()
            
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # milestones: [20%, 40%, 40%] [60, 120, 160]
        milestones = [int(num_epochs * 0.3), int(num_epochs * 0.6)]
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5) #learning rate decay
        
        for epoch in range(num_epochs):
            
            train_loss = 0.0
            val_loss = 0.0
            train_correct = 0
            train_total = 0
            
            with tqdm(total=(len(train_loader))) as _tqdm:#总长度是data的长度
                _tqdm.set_description(
                    '[Training] Epoch [{:03}/{:03}] Lr [{:.1e}]'.format(epoch + 1, num_epochs, optimizer.param_groups[0]['lr'],)
                )#前缀设置一些想要的更新信息
                self.model.train()
                for i, (inputs, labels) in enumerate(train_loader):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    outputs = self.model(inputs)
                    self.before_loss(criterion, inputs, labels)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()

                    _tqdm.set_postfix(
                        Loss="{:.3f}".format(train_loss / train_total),
                        Acc="{:.2f}%".format(100 * train_correct / train_total)
                    )
                    _tqdm.update(1)#更新步长为一个batchsize长度
                    
                self.after_inference()

            self.model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                with tqdm(total=(len(val_loader))) as _tqdm:#总长度是data的长度
                    _tqdm.set_description(
                        '[Testing ] Epoch [{:03}/{:03}] Lr [{:.1e}]'.format(epoch + 1, num_epochs, optimizer.param_groups[0]['lr'],)
                    )#前缀设置一些想要的更新信息
                    for i, (inputs, labels) in enumerate(val_loader):
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        outputs = self.model(inputs)

                        self.before_loss(criterion, inputs, labels)
                        loss = criterion(outputs, labels)

                        val_loss += loss.item() * inputs.size(0)
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()

                        _tqdm.set_postfix(
                            Loss="{:.3f}".format(val_loss / val_total),
                            Acc="{:.2f}%".format(100 * val_correct / val_total)
                        )
                        _tqdm.update(1)

            # train_loss /= len(train_loader.dataset)
            # train_accuracy = train_correct / train_total
            # val_loss /= len(val_loader.dataset)
            # val_accuracy = val_correct / val_total
            # print("Epoch {:>2} - Train Loss: {:.4f} - Val Loss: {:.4f} - Train Accuracy: {:.2f}% - Val Accuracy: {:.2f}%".format(
            #     epoch+1, train_loss, val_loss, train_accuracy*100, val_accuracy*100))
            train_scheduler.step()
            print()

    def eval(self, test_loader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)

        correct = 0
        total = 0

        self.model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print("Test Accuracy: {:.2f}%".format(accuracy * 100))
