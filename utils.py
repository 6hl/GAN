import os
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

from models import *

class trainer(object):
    def __init__(self, epochs=5, in_size=100, batch_size=128, lr=0.0002, beta=0.5, crop_shape=64, path=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.in_size = in_size
        self.lr = lr
        self.beta = beta
        self.crop_shape = crop_shape
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_data(path)
    
    def _config(self):
        self.cwd = os.getcwd()
        results_path = os.path.join(self.cwd, "results")
        if not os.path.isdir(results_path):
            os.mkdir(results_path)

        self.image_path = os.path.join(results_path, self.name)
        if os.path.isdir(self.image_path):
            print(f"Results stored in {self.image_path}")
        else:
            os.mkdir(self.image_path)
            print(f"Results stored in {self.image_path}")
        self.image_num = 0

    def load_data(self, path=None):
        # airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks
        dataset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=True, 
            download=True, 
            transform=transforms.Compose([
                transforms.Resize(self.crop_shape),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
        data = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size,
            shuffle=True
        )
        self.dataset = data
    
    def make_model(self):
        self.gen = Generator().to(self.device)
        self.disc = Discriminator(model=self.name).to(self.device)
        self.gen.apply(self._init_model)
        self.disc.apply(self._init_model)
        self._init_training()

    def _init_model(self, model):
        if model.__class__.__name__.find("Conv") != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
        elif model.__class__.__name__.find("BatchNorm") != -1:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
            nn.init.constant_(model.bias.data, 0)

    def _init_training(self, g_lr=None, d_lr=None):
        if not (g_lr and d_lr):
            g_lr = self.lr
            d_lr = self.lr

        self.test = torch.randn(10, self.in_size, 1, 1, device=self.device)
        self.criterion = nn.BCELoss()
        if self.name == "ACGAN":
            self.criterion2 = nn.NLLLoss()

        self.labels = {
            "real": 1,
            "fake": 0
        }

        self.gopt = torch.optim.Adam(
            self.gen.parameters(), 
            lr=g_lr, 
            betas=(self.beta, 0.999)
        )
        
        self.dopt = torch.optim.Adam(
            self.disc.parameters(), 
            lr=d_lr, 
            betas=(self.beta, 0.999)
        )

    def save_test(self, train_num):
        save_path = os.path.join(self.image_path, f"Epoch_{train_num+1}")
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        with torch.no_grad():
            test_generated = self.gen(self.test).detach().cpu()
            for i in range(self.test.shape[0]):
                vutils.save_image(
                    test_generated[i], 
                    fp=os.path.join(save_path, f"{train_num+1}_generator_image_{i}.jpg"), 
                    normalize=True
                )
            plt.imshow(np.transpose(vutils.make_grid(test_generated, padding=2, normalize=True), (1,2,0)))
            plt.savefig(os.path.join(save_path, f"{train_num+1}_all_images.jpg"))

    def train(self):
        pass


class DCGAN(trainer):
    def __init__(self, *args, **kwargs):
        super(DCGAN, self).__init__(*args, **kwargs)
        self.name = "DCGAN"
        self._config()
        self.make_model()
        self.train()
    
    def train(self):
        self.save_imgs = []
        skip = len(self.dataset) -1
        start_time = perf_counter()
        for e in range(self.epochs):
            for idx, samples in enumerate(self.dataset):
                self.disc.zero_grad()

                samples = samples[0].to(self.device)
                labels = torch.full((samples.size(0),), self.labels["real"], dtype=torch.float).to(self.device)
                
                predicted_label = self.disc(samples)
                loss1 = self.criterion(predicted_label.view(-1), labels)
                loss1.backward()

                noise = torch.randn(samples.size(0), self.in_size, 1, 1, device=self.device)
                gen_fake = self.gen(noise)
                labels.fill_(self.labels["fake"])

                predict_fake = self.disc(gen_fake.detach())
                loss2 = self.criterion(predict_fake.view(-1), labels)
                loss2.backward()

                tot_loss = loss1.mean().item() + loss2.mean().item()
                # tot_loss.backward()
                self.dopt.step()

                self.gen.zero_grad()
                labels.fill_(self.labels["real"])
                predict = self.disc(gen_fake)
                loss3 = self.criterion(predict.view(-1), labels)
                loss3.backward()
                self.gopt.step()
            
            if tot_loss > 0.4 and tot_loss < 0.6:
            # if (e+1) % 20 == 0:
                print(f"Epoch: {e+1}, Loss Disc: {tot_loss:.4f}, Loss Gen: {loss3.item():.4f}, Time: {perf_counter()-start_time:.4f}")
                self.save_test(train_num=e)
                start_time = perf_counter()

class WGAN(trainer):
    def __init__(self, clip_value=0.01, n_critic=5, iterations=40000, *args, **kwargs):
        super(WGAN, self).__init__(*args, **kwargs)
        self.name = "WGAN"
        self.clip_value = clip_value
        self.n_critic = n_critic
        self.iterations = iterations
        self._config()
        self.make_model()
                
        self.gopt = torch.optim.RMSprop(
            self.gen.parameters(), 
            lr=self.lr
        )
        
        self.dopt = torch.optim.RMSprop(
            self.disc.parameters(), 
            lr=self.lr
        )

        self.train()
    
    def train(self):
        self.save_imgs = []        
        start_time = perf_counter()

        for it in range(self.iterations):
            for n in range(self.n_critic):
                self.disc.zero_grad()

                samples = next(iter(self.dataset))
                samples = samples[0].to(self.device)
                labels = torch.full((samples.size(0),), self.labels["real"], dtype=torch.float).to(self.device)
                
                predict_real = self.disc(samples)
                real_loss = predict_real.mean()
                
                noise = torch.randn(samples.size(0), self.in_size, 1, 1, device=self.device)
                gen_fake = self.gen(noise)
                labels.fill_(self.labels["fake"])

                predict_fake = self.disc(gen_fake.detach())
                fake_loss = predict_fake.mean()

                aggre_loss = -real_loss + fake_loss
                aggre_loss.backward()

                self.dopt.step()
                for p in self.disc.parameters():
                    p.data.clamp_(-self.clip_value, self.clip_value)

            self.gen.zero_grad()
            labels.fill_(self.labels["real"])
            predict = self.disc(gen_fake)
            gen_loss = -predict.mean() #Neg?
            gen_loss.backward()
            self.gopt.step()

            if (it+1) % 200 == 0:
                print(f"Iteration: {it+1}, Loss Disc: {aggre_loss.item():.4f}, Loss Gen: {gen_loss.item():.4f}, Time: {perf_counter()-start_time:.1f}")
                start_time = perf_counter()
                self.save_test(train_num=it)

class ACGAN(trainer):
    def __init__(self, *args, **kwargs):
        super(ACGAN, self).__init__(*args, **kwargs)
        self.name = "ACGAN"
        self.num_classes = 10
        self._config()
        self.make_model()
        self.train()
        torch.save(self.gen.state_dict(), f"{self.name}_generator.pth")
    
    def make_model(self):
        self.gen = ACGenerator().to(self.device)
        self.disc = ACDiscriminator().to(self.device)
        self.gen.apply(self._init_model)
        self.disc.apply(self._init_model)
        self._init_training()
        # self._init_training(g_lr=0.0001, d_lr=0.0004)

    def _accuracy(self, preds, labels):
        preds_ = preds.data.max(1)[1]
        correct = preds_.eq(labels.data).cpu().sum()
        acc = float(correct) / float(len(labels.data)) * 100.0
        return acc
    
    def train(self):
        self.save_imgs = []
        start_time = perf_counter()
        for e in range(self.epochs):
            for idx, (samples, classes) in enumerate(self.dataset):
                self.disc.zero_grad()

                samples, classes = samples.to(self.device), classes.to(self.device)
                labels = torch.full((classes.size(0),), self.labels["real"], dtype=torch.float).to(self.device)
                
                pred_label, pred_class = self.disc(samples)
                pred_loss = self.criterion(pred_label.view(-1), labels)

                class_loss = self.criterion2(pred_class, classes)
                tot_real_loss = pred_loss + class_loss
                tot_real_loss.backward()

                noise = torch.randn(samples.shape[0], self.in_size, 1, 1, device=self.device)
                fake_classes = torch.randint(low=0, high=self.num_classes, size=(classes.shape), device=self.device)
                fake_samples = self.gen(noise)
                labels.fill_(self.labels["fake"])

                pred_fake_label, pred_fake_class = self.disc(fake_samples.detach())
                pred_loss_fake = self.criterion(pred_fake_label.view(-1), labels)
                class_loss = self.criterion2(pred_fake_class, fake_classes)
                tot_fake_loss = pred_loss_fake + class_loss
                tot_fake_loss.backward()
                self.dopt.step()

                self.gen.zero_grad()
                labels.fill_(self.labels["real"])
                predict, fake_class = self.disc(fake_samples)
                gen_pred_loss = self.criterion(predict.view(-1), labels)
                gen_class_loss = self.criterion2(fake_class, classes)
                gen_loss = gen_pred_loss + gen_class_loss
                gen_loss.backward()
                self.gopt.step()

                acc = self._accuracy(pred_class, classes)
            tot_disc_loss = pred_loss.item()+pred_loss_fake.item()
            if  tot_disc_loss> 0.4 and tot_disc_loss < 0.6 and (e+1)>50:
            # if (e+1) % 50 == 0:
                print(f"Epoch: {e+1}, Loss Disc: {tot_disc_loss:.4f}, Loss Gen: {gen_loss.item():.4f}, Accuracy: {acc:0.2f}, Time: {perf_counter()-start_time:.1f}")
                self.save_test(train_num=e)
                start_time = perf_counter()