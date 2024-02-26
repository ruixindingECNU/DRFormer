from __future__ import print_function
import torch
import torch.optim as optim
import copy
import math

import torch.nn.init as init

def str2bool(str):
    return True if str.lower() == 'true' else False

def add_sparse_args(parser):
    parser.add_argument('--sparse', type=str2bool, default=True, help='Enable sparse mode. Default: True.')
    parser.add_argument('--fix', type=str2bool, default=False, help='Fix the mask during training. Default: True.')
    parser.add_argument('--death-rate', type=float, default=0.5, help='The pruning rate / death rate.')
    parser.add_argument('--update_frequency', type=float, default=0.3, metavar='N', help='how many iterations to train between parameter exploration')
    parser.add_argument('--group_num', type=int, default=8, help='The dynamic linear layer is divided into mutiple groups.')
    parser.add_argument('--learnable_mask_epoches', type=float, default=0.5, help='The number of epoches that the sparse mask is learnable.')
    parser.add_argument('--sparsity', type=float, default=0.3, help='The activation ratio of the dynamic linear layer.')
    parser.add_argument('--death_mode', type=str, default='magnitude', help='The mode of deactivation. Choose from: magnitude, Taylor_FO.')
    
class CosineDecay(object):
    def __init__(self, death_rate, T_max, eta_min=0.001, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self):
        return self.sgd.param_groups[0]['lr']


class Masking(object):
    def __init__(self, optimizer, death_rate=0.3, death_rate_decay=None, train_loader=None, args=None):

        self.args = args
        self.death_rate_decay = death_rate_decay

        self.optimizer = optimizer

        self.death_rate = death_rate
        self.steps = 0
        self.explore_step = 0
        self.w_ratio = 0

        self.decay_flag = True

        self.total = 0
        self.nonzeros = 0
        self.zeros = 0
        self.loader = train_loader
        
        self.remove_idx_list = []
        self.grow_idx_list = []
                

    def step(self):
        self.optimizer.step()
        self.apply_mask()
        # print(self.weight.data[-1].cpu().numpy().tolist())

        if self.decay_flag:
            self.death_rate_decay.step()
            self.death_rate = self.death_rate_decay.get_dr()
        else:
            self.death_rate = 0


    def add_module(self, module):
        self.module = module
        for name, tensor in module.named_parameters():
            if  name == 'masked_linear.weight':
                self.weight_name = 'masked_linear.weight'
                self.weight = tensor
                self.out_features = tensor.size()[0]
                self.in_features = tensor.size()[1]
                self.step_in_features = math.floor(self.in_features/self.args.group_num)
                self.step_out_features = math.floor(self.out_features/self.args.group_num)
                self.mask = torch.Tensor()
                self.window_size = []
                print("self.args.sparsity: " + str(self.args.sparsity))
                for i in range(1,self.args.group_num+1,1):
                    self.mask = torch.cat((self.mask,torch.cat((torch.full((self.step_out_features,self.in_features - math.ceil((i * self.step_in_features) * self.args.sparsity)),0), torch.full((self.step_out_features,math.ceil((i * self.step_in_features) * self.args.sparsity)),1)),dim=-1)), dim=-2)
                    self.window_size.extend([i * self.step_in_features] * self.step_out_features)
                self.mask = self.mask.cuda()
                self.apply_mask()
                self.cal_nonzero_counts()
                print('total_size: '+str(self.total))
                print('sparse_size: '+str(self.w_ratio))
        

    def cal_nonzero_counts(self):
        self.total = self.mask.numel()
        self.nonzeros = self.mask.sum().item()
        self.zeros = self.total - self.nonzeros
        w_nonzeros = (self.weight.data != 0).sum().item()
        w_numele = self.weight.data.nelement()
        self.w_ratio = w_nonzeros / w_numele
        


    def apply_mask(self):
        self.weight.data = self.weight.data*self.mask
        
        
                    
    def truncate_weights_channel(self, total_score=None):
        
        self.cal_nonzero_counts()
        
        # death
        if self.args.death_mode == 'Taylor_FO':
            new_mask, remove_list, self.remove_idx_list = self.taylor_FO(self.mask, total_score, group_num=self.args.group_num)
        else:
            new_mask, remove_list, self.remove_idx_list = self.magnitude_death_channel(self.mask, self.weight, group_num=self.args.group_num)

        coincidence_rate_list = []
        if (len(self.grow_idx_list) > 0):
            for i, remove_idx in enumerate(self.remove_idx_list): 
                grow_idx = self.grow_idx_list[i]
                set_c = set(remove_idx) & set(grow_idx)
                list_c = list(set_c)
                if(len(remove_idx) == 0):
                    coincidence_rate = 1
                else:
                    coincidence_rate = len(list_c)/len(remove_idx)
                coincidence_rate_list.append(coincidence_rate)
                
            print(coincidence_rate_list)

        self.mask[:] = new_mask

        new_mask = self.mask.data.byte()

        # growth
        new_mask, self.grow_idx_list = self.random_growth_channel(new_mask, self.weight, group_num=self.args.group_num, num_rm_list=remove_list)        

        self.mask = new_mask.float()

        self.regrowed_mask = new_mask.float()

        self.apply_mask()
        
        weight_chunk = torch.chunk(self.weight, self.args.group_num, dim=0)
        mask_chunk = torch.chunk(self.mask, self.args.group_num, dim=0)
        
        for i in range(0, len(weight_chunk)):
            grow_idx = self.grow_idx_list[i]
            grow_weight = torch.empty(weight_chunk[i].shape).cuda()
            grow_mask = torch.zeros_like(grow_weight, dtype=torch.float32).cuda()
            grow_mask.view(-1)[grow_idx] = 1
            
            init.kaiming_uniform_(grow_weight, a=math.sqrt(5))
            grow_weight = grow_weight.view(-1)
            weight_chunk[i].data = (weight_chunk[i].view(-1)+grow_weight*grow_mask.view(-1)).view(self.step_out_features,-1).data
        weight_list = torch.cat(weight_chunk,dim=0)
        self.weight.data = weight_list.data

        
    def taylor_FO(self,mask,total_score,group_num=8):
        mask_chunk = list(torch.chunk(self.mask, group_num, dim=0))
        total_score_chunk = torch.chunk(total_score, group_num, dim=0)
        mask_list = []
        remove_list = []
        remove_idx_list = []
        window_size_chunk = list(torch.chunk(torch.Tensor(self.window_size), group_num, dim=0))
        for i, w_c in enumerate(total_score_chunk):
            num_nonzeros = (mask_chunk[i] != 0).sum().item()
            num_zeros = (mask_chunk[i] == 0).sum().item()
            window_size_group = window_size_chunk[i]
            num_remove = math.ceil(self.death_rate * num_nonzeros)
            

            old_weight = torch.nonzero(torch.chunk(mask, group_num, dim=0)[i].view(-1)==0).squeeze()
            
            a = torch.zeros_like(w_c)
            a[:,:]=1000
            a = a * (torch.ones_like(mask_chunk[i])-mask_chunk[i])
            w_c.data = w_c.data + a
            
            x, idx = torch.sort(w_c.data.view(-1))
            k = math.ceil(num_zeros + num_remove)
            threshold = x[-k].item()
            remove_idx = idx[-k:]
            
            remove_idx_list.append(list(set(torch.sort(remove_idx)[0].cpu().numpy().tolist()) - set(torch.sort(old_weight)[0].cpu().numpy().tolist())))

            mask_list.append(w_c.data < threshold)
            remove_list.append(num_remove)

        mask_new = torch.cat(tuple(mask_list), 0)
        return mask_new, remove_list, remove_idx_list    


    def magnitude_death_channel(self, mask, weight, group_num=8):

        weight_chunk = torch.chunk(weight, group_num, dim=0)
        mask_chunk = list(torch.chunk(mask, group_num, dim=0))
        mask_list = []
        remove_list = []
        remove_idx_list = []
        for i, w_c in enumerate(weight_chunk):
            num_nonzeros = (mask_chunk[i] != 0).sum().item()
            num_zeros = (mask_chunk[i] == 0).sum().item()
            num_remove = math.ceil(self.death_rate * num_nonzeros)

            old_weight = torch.nonzero(torch.chunk(mask, group_num, dim=0)[i].view(-1)==0).squeeze()
            x, idx = torch.sort(torch.abs(w_c.data.view(-1)))
            k = math.ceil(num_zeros + num_remove)
            threshold = x[k - 1].item()
            remove_idx = idx[:k]
            remove_idx_list.append(list(set(torch.sort(remove_idx)[0].cpu().numpy().tolist()) - set(torch.sort(old_weight)[0].cpu().numpy().tolist())))

            mask_list.append(torch.abs(w_c.data) > threshold)
            
            remove_list.append(num_remove)

        mask_new = torch.cat(tuple(mask_list), 0)
        return mask_new, remove_list, remove_idx_list


    def random_growth_channel(self, mask, weight, group_num=8, num_rm_list=None):

        # k_s = [9, 19, 39]
        weight_chunk = torch.chunk(weight, group_num, dim=0)
        mask_chunk = list(torch.chunk(mask, group_num, dim=0))
        window_size_chunk = list(torch.chunk(torch.Tensor(self.window_size), group_num, dim=0))
        num_rm_list = num_rm_list
        mask_list = []
        grow_idx_list = []
        for i, w_c in enumerate(weight_chunk):
            num_remove = num_rm_list[i]
            
            num_remain = num_remove + (mask_chunk[i] != 0).sum().item()
            window_size_group = window_size_chunk[i]
            mask_group = mask_chunk[i]

            weight_new = copy.deepcopy(w_c.data) + 20.
            weight_new = weight_new * mask_chunk[i]
            old_weight = torch.nonzero(weight_new.view(-1)>0).squeeze()

            x = torch.rand(mask_group.shape).cuda()
            for j, w in enumerate(w_c):
                ind = list(range(len(mask_group[j])-min(window_size_group[j].int().item(), len(mask_group[j])),len(mask_group[j])))
                x[j, ind] += 10.0

            weight_add = weight_new + x

            new_mask = torch.zeros_like(weight_add, dtype=torch.float32).cuda()
            y, idx = torch.sort(torch.abs(weight_add).flatten(), descending=True)  ## big to small
            new_mask.data.view(-1)[idx[:num_remain]] = 1.0
            grow_idx_list.append(list(set(idx[:num_remain].cpu().numpy().tolist()) - set(old_weight.cpu().numpy().tolist())))

            mask_list.append(new_mask)

        mask_new = torch.cat(tuple(mask_list), 0)
        return mask_new, grow_idx_list



    ### add function
    def death_decay_update(self, decay_flag=True):
        self.decay_flag = decay_flag