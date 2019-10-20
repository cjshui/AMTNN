"""
This version considers task's datasets have equal number of labeled samples
"""

import os
import json
from collections import defaultdict
import numpy as np

from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

import util
from util import in_feature_size
import alpha_opt
import data_loading as db
from torch.optim import lr_scheduler

class MTL_pairwise(object):
    def __init__(self, ft_extrctor_prp, hypoth_prp, discrm_prp,  **kwargs):
        final_results = defaultdict()


        # ######################### argument definition ###############
        self.criterion = kwargs ['criterion']
        self.c3_value = kwargs['c3']
        self.grad_weight = kwargs['grad_weight']
        self.img_size = kwargs['img_size']
        self.num_chnnl = kwargs['chnnl']
        self.lr = kwargs['lr']
        self.momentum = kwargs['momentum']
        self.epochs = kwargs['epochs']
        num_tr_smpl = kwargs['tr_smpl']
        num_test_smpl = kwargs['test_smpl']
        self.trial = kwargs['Trials']
        self.tsklist = kwargs['tsk_list']
        self.num_tsk = len(self.tsklist)

        if self.criterion=='wasserstien': self.stp_sz_sch = 30
        else: self.stp_sz_sch = 50

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.alpha = np.ones((self.num_tsk, self.num_tsk)) * (0.1 / (self.num_tsk - 1))
        np.fill_diagonal(self.alpha, 0.9)

        self.wrdir = os.path.join(os.getcwd(), '_'.join( self.tsklist)+'_'+str(num_tr_smpl)+'_'+ str(self.epochs)+'_'+self.criterion, 'runs_'+str(self.c3_value))
        try:
            os.makedirs(self.wrdir)
        except OSError:
            if not os.path.isdir(self.wrdir):
                raise

        with open(os.path.join(self.wrdir, 'info_itr_'+str(self.trial)+'.json'), 'a') as outfile:
            json.dump([ft_extrctor_prp,hypoth_prp,discrm_prp], outfile)
            json.dump(kwargs, outfile)


        # Constructing F -> H and F -> D
        self.FE = util.feature_extractor(ft_extrctor_prp).construct().to(self.device)
        print (self.FE)

        self.hypothesis = [util.classifier(hypoth_prp).to(self.device) for _ in range(self.num_tsk)]
        print (self.hypothesis[0])


        self.discrm = {'{}{}'.format(i, j): util.classifier(discrm_prp).to(self.device)for i in range(self.num_tsk) for
                       j in range(i + 1, self.num_tsk)}
        print (self.discrm['01'])
        all_parameters_h = sum([list(h.parameters()) for h in self.hypothesis], [])
        all_parameters_discrm = sum([list(self.discrm[d].parameters()) for d in self.discrm], [])


        self.optimizer = optim.SGD(list(self.FE.parameters()) + list(all_parameters_h) + list(all_parameters_discrm),
                                       lr=self.lr,
                                       momentum=self.momentum)

        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.stp_sz_sch, gamma=0.5)

        train_loader, test_loader, validation_loader = db.data_loading(self.img_size,  num_tr_smpl,num_test_smpl, self.tsklist )

        self.writer = SummaryWriter(os.path.join(self.wrdir, 'itr'+str(self.trial)))

        Total_loss = []

        for epoch in range(self.epochs):
            self.scheduler.step(epoch)
            whole_loss  =  self.model_fit(train_loader, epoch)
            Total_loss.append(whole_loss)
            tasks_trAcc = self.model_eval(train_loader, epoch, 'train')
            tasks_valAcc = self.model_eval(validation_loader, epoch, 'validation')
            tasks_teAcc = self.model_eval(test_loader, epoch, 'test')

            # if np.abs(np.mean(Total_loss[-5:-1]) - Total_loss[-1]) < 0.002 :
            #     print('Stop learning, reach to a stable point at epoch {:d} with total loss {:.4f}'.format(epoch,
            #                                                                                       Total_loss[-1]))
            #     break
            if 1.5*np.mean(Total_loss[-5:-1]) < Total_loss[-1]:

                print ('****** Increasing of training error')
                break

            final_results['alpha_c3_'+str(self.c3_value)] = (self.alpha).tolist()
            final_results['Tasks_val_Acc_c3_'+str(self.c3_value)] = (tasks_valAcc).tolist()
            final_results['Tasks_test_Acc_c3_' + str(self.c3_value) ] = (tasks_teAcc).tolist()
            final_results['Tasks_train_Acc_c3_'+str(self.c3_value)] = (tasks_trAcc).tolist()

        with open(os.path.join(self.wrdir, 'info_itr_'+str(self.trial)+'.json'), 'a') as outfile:
           json.dump(final_results, outfile)

        final_prmtr = defaultdict()
        final_prmtr['FE'] = self.FE.state_dict()
        for i,h in enumerate(self.hypothesis):
            final_prmtr['hypo'+str(i)] = h.state_dict()
        for k, D in self.discrm.items():
            final_prmtr['dicrm'+k] = D.state_dict()

        torch.save(final_prmtr, os.path.join(self.wrdir, 'itr'+str(self.trial),'MTL_parameters.pt'))
        self.writer.close()





    def model_fit(self, data_loader, epoch):

        discrm_distnc_mtrx = np.zeros((self.num_tsk, self.num_tsk))
        loss_mtrx_hypo_vlue = np.zeros((self.num_tsk, self.num_tsk))
        weigh_loss_hypo_vlue, correct_hypo = np.zeros(self.num_tsk), np.zeros(self.num_tsk)
        Total_loss = 0
        n_batch = 0

        # set train mode
        self.FE.train()
        for t in range(self.num_tsk):
            self.hypothesis[t].train()
            for j in range(t + 1, self.num_tsk):
                self.discrm['{}{}'.format(t, j)].train()
        # #####
        for tasks_batch in zip(*data_loader):
            Loss_1, Loss_2 = 0, 0
            n_batch += 1
            # data = (x,y)
            inputs = torch.cat([batch[0] for batch in tasks_batch])


            btch_sz = len(tasks_batch[0][0])
            targets = torch.cat([batch[1] for batch in tasks_batch])

            # inputs = (x1,...,xT)  targets = (y1,...,yT)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            features = self.FE(inputs)
            features = features.view(features.size(0), -1)
            for t in range(self.num_tsk):
                w = torch.tensor([np.tile(self.alpha[t, i], reps=len(data[0])) for i, data in enumerate(tasks_batch)],
                                 dtype=torch.float).view(-1)
                w = w.to(self.device)

                label_prob = self.hypothesis[t](features)

                pred = label_prob[t * (btch_sz):(t + 1) * btch_sz].argmax(dim=1, keepdim=True)
                correct_hypo[t] += (
                (pred.eq(targets[t * btch_sz:(t + 1) * btch_sz].view_as(pred)).sum().item()) / btch_sz)

                hypo_loss = torch.mean(w * F.cross_entropy(label_prob, targets, reduction='none'))

                # definition of loss to be optimized
                Loss_1 += hypo_loss
                weigh_loss_hypo_vlue[t] += hypo_loss.item()

                loss_mtrx_hypo_vlue[t, :] += [F.cross_entropy(label_prob[j * (btch_sz):(j + 1) * btch_sz, :],
                                                              targets[j * (btch_sz):(j + 1) * btch_sz],
                                                              reduction='mean').item() for j in range(self.num_tsk)]

                for k in range(t + 1, self.num_tsk):
                    # w = (alpha_{tk}+alpha_{kt}) assumption: matrix alpha is not symmetric
                    alpha_domain = torch.tensor(self.alpha[t, k] + self.alpha[k, t], dtype=torch.float)
                    alpha_domain = alpha_domain.to(self.device)
                    if self.criterion =='h_divergence':
                        domain_y = torch.cat([torch.ones(len(tasks_batch[t][0]), dtype=torch.float),
                                              torch.zeros(len(tasks_batch[k][0]), dtype=torch.float)])
                        # domain_x = torch.cat([tasks_batch[t-1][0], tasks_batch[k-1][0] ])
                        domain_y = domain_y.to(self.device)
                        domain_features = torch.cat([features[t * btch_sz:(t + 1) * btch_sz], features[k * btch_sz:(k + 1) * btch_sz]])

                        domain_features = domain_features.view(domain_features.size(0), -1)

                        domain_pred = self.discrm['{}{}'.format(t, k)](domain_features).squeeze()

                        disc_loss = F.binary_cross_entropy(domain_pred, domain_y)

                        #  discriminator accuracy defines H-divergence
                        domain_lbl = domain_pred >= 0.5
                        domain_lbl = domain_lbl.type(torch.cuda.FloatTensor)
                        discrm_distnc_mtrx[t, k] += (domain_lbl.eq(domain_y).sum().item()) / len(domain_y)
                        discrm_distnc_mtrx[k, t] = discrm_distnc_mtrx[t, k]

                        print(discrm_distnc_mtrx[t, :])

                    elif self.criterion =='wasserstien':

                        features_t = features[t * btch_sz:(t + 1) * btch_sz]
                        features_t = features_t.view(features_t.size(0), -1)
                        features_k = features[k * btch_sz:(k + 1) * btch_sz]
                        features_k = features_k.view(features_k.size(0), -1)
                        pred_k = self.discrm['{}{}'.format(t, k)](features_k).squeeze()
                        pred_t = self.discrm['{}{}'.format(t, k)](features_t).squeeze()

                        gradient_pntly=self.gradient_penalty(inputs[t * btch_sz:(t + 1) * btch_sz],inputs[k * btch_sz:(k + 1) * btch_sz], t, k)
                        # critic loss --->  E(f(x)) - E(f(y)) + gamma* ||grad(f(x+y/2))-1||
                        disc_loss = (pred_t.mean() - pred_k.mean() ) + self.grad_weight *gradient_pntly
                        #  negative sign compute wasserstien distance
                        discrm_distnc_mtrx[t, k] += -(pred_t.mean() - pred_k.mean()).item()
                        discrm_distnc_mtrx[k, t] = discrm_distnc_mtrx[t, k]

                    disc_loss = alpha_domain * disc_loss
                    Loss_2 += disc_loss


            if n_batch % 500 == 0:
                grid_img = torchvision.utils.make_grid(inputs, nrow=5, padding=30)
                self.writer.add_image('result Image', grid_img)

            Loss = torch.mean(Loss_1) + Loss_2 * (1 / self.num_tsk)
            Total_loss += Loss.item()

            # loss formula for all tasks regarding the current batch
            self.optimizer.zero_grad()
            Loss.backward()
            self.optimizer.step()


        discrm_distnc_mtrx /= n_batch
        weigh_loss_hypo_vlue /= n_batch
        loss_mtrx_hypo_vlue /= n_batch
        correct_hypo /= n_batch
        Total_loss /= n_batch


        print('================== epoch {:d} ========'.format(epoch))
        print('Final Total Loss {:.3f}'.format(Total_loss ))
        print('discriminator distance based on '+self.criterion +'\n'+ str(discrm_distnc_mtrx))

        print(' hypothesis loss \n' + str(loss_mtrx_hypo_vlue))
        print(' hypothesis accuracy \n' + str(correct_hypo * 100))
        print('coefficient:',self.alpha)
        self.writer.add_scalars('MTL_total_loss', {'MTL_total_loss': Total_loss}, epoch)
        for t in range(self.num_tsk):
            # self.writer.add_scalars('task_' + str(t) + '/loss', {'loss_train': loss_mtrx_hypo_vlue[t, t]}, epoch)
            for j in range(self.num_tsk):
                if j != t:
                    self.writer.add_scalars('task_' + str(t) + '/Discrm_distance',
                                            {'loss_D' + '_'.join([self.tsklist[t],self.tsklist[j]]): discrm_distnc_mtrx[t, j]}, epoch)
                self.writer.add_scalars('task_' + str(t) + '/alpha',
                                        {'alpha' + '_'.join([self.tsklist[t],self.tsklist[j]]): self.alpha[t, j]}, epoch)

        if epoch % 1 == 0:
            c_2, c_3 = 1 * np.ones(self.num_tsk), self.c3_value * np.ones(self.num_tsk)
            self.alpha = alpha_opt.min_alphacvx(self.alpha.T, c_2, c_3, loss_mtrx_hypo_vlue.T, discrm_distnc_mtrx.T)

            self.alpha = self.alpha.T
        return  Total_loss

    def model_eval(self, data_loader, epoch, phase='test'):

        loss_hypo_vlue = np.zeros(self.num_tsk)
        correct_hypo = np.zeros(self.num_tsk)
        self.FE.eval()
        for t in range(self.num_tsk):
            n_batch_t = 0
            self.hypothesis[t].eval()
            for j in range(t + 1, self.num_tsk):
                self.discrm['{}{}'.format(t, j)].eval()

            for inputs, targets in (data_loader[t]):
                n_batch_t += 1
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                features = self.FE(inputs)
                features = features.view(features.size(0), -1)
                label_prob = self.hypothesis[t](features)
                pred = label_prob.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct_hypo[t] += ((pred.eq(targets.view_as(pred)).sum().item()) / len(pred))
                loss_hypo_vlue[t] += F.cross_entropy(label_prob, targets, reduction='mean').item()
                if n_batch_t % 100 == 0:
                    grid_img = torchvision.utils.make_grid(inputs, nrow=5, padding=30)
                    self.writer.add_image('result Image_' + phase, grid_img)

            loss_hypo_vlue[t] /= n_batch_t
            correct_hypo[t] /= n_batch_t


            self.writer.add_scalars('task_' + str(t) + '/loss', {'loss_' + phase: loss_hypo_vlue[t]}, epoch)
            self.writer.add_scalars('task_' + str(t) + '/Acc', {'Acc_' + phase: correct_hypo[t]}, epoch)


        print('\t === hypothesiz **' + phase + '** loss \n' + str(loss_hypo_vlue))
        print('\t === hypothesiz **' + phase + '** accuracy \n' + str(correct_hypo * 100))

        return correct_hypo

    def gradient_penalty(self, data_t, data_k, t, k):
        batch_size = data_k.size()[0]
        # Calculate interpolation
        theta = torch.rand(batch_size, 1, 1,1)
        theta = theta.expand_as(data_t)
        theta = theta.to(self.device)
        interpolated = theta * data_t + (1 - theta) * data_k

        # computing gradient w.r.t interplated sample
        interpolated = Variable(interpolated, requires_grad=True)

        interpolated = interpolated.to(self.device)

        features_intrpltd = self.FE(interpolated)
        features_intrpltd = features_intrpltd.view(features_intrpltd.size(0), -1)
        # Calculate probability of interpolated examples
        prob_interpolated = self.discrm['{}{}'.format(t, k)](features_intrpltd).squeeze()

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return  ((gradients_norm - 1) ** 2).mean()


def main():
    """"options for criterion is wasserstien, h_divergence"""
    # criterion = ['wasserstien', 'h_divergence']
    itertn = 1

    # for c3_value in [0.5, 0.2, 1]:
    c3_value = 0.5
    for trial in range(1):
            args = {'img_size': 28,
                    'chnnl': 1,
                    'lr': 0.01,
                    'momentum': 0.9,
                    'epochs': 1,
                    'tr_smpl': 1000,
                    'test_smpl': 10000,
                    'tsk_list': ['mnist', 'svhn', 'm_mnist'],
                    'grad_weight': 1,
                    'Trials': trial,
                    #'criterion': 'h_divergence',
                    'criterion': 'wasserstien',
                    'c3':c3_value}
            ft_extrctor_prp = {'layer1': {'conv': [1, 32, 5, 1, 2], 'elu': [], 'maxpool': [3, 2, 0]},
                               'layer2': {'conv': [32, 64, 5, 1, 2], 'elu': [], 'maxpool': [3, 2, 0]}}

            hypoth_prp = {
                'layer3': {'fc': [util.in_feature_size(ft_extrctor_prp, args['img_size']), 128], 'act_fn': 'elu'},
                'layer4': {'fc': [128, 10], 'act_fn': 'softmax'}}

            discrm_prp = {'reverse_gradient': {},
                          'layer3': {'fc': [util.in_feature_size(ft_extrctor_prp, args['img_size']), 128],
                                     'act_fn': 'elu'},
                          'layer4': {'fc': [128, 1], 'act_fn': 'sigm'}}

            mtl = MTL_pairwise(ft_extrctor_prp, hypoth_prp, discrm_prp, **args)
            del mtl


if __name__ == '__main__':
    main()



