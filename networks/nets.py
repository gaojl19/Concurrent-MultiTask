from importlib_metadata import distribution
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networks.init as init
import copy


class ZeroNet(nn.Module):
    def forward(self, x):
        return torch.zeros(1)


class Net(nn.Module):
    def __init__(
            self, output_shape,
            base_type,
            append_hidden_shapes=[],
            append_hidden_init_func=init.basic_init,
            net_last_init_func=init.uniform_init,
            activation_func=F.relu,
            add_bn=False,
            add_ln=False,
            **kwargs):

        super().__init__()

        self.base = base_type(activation_func=activation_func, **kwargs)
        self.activation_func = activation_func
        append_input_shape = self.base.output_shape
        self.append_fcs = []
        for i, next_shape in enumerate(append_hidden_shapes):
            fc = nn.Linear(append_input_shape, next_shape)
            append_hidden_init_func(fc)
            if add_bn:
                module = nn.Sequential(
                    nn.BatchNorm1d(append_input_shape),
                    fc,
                    nn.BatchNorm1d(next_shape)
                )
            elif add_ln:
                module = nn.Sequential(
                    nn.LayerNorm(append_input_shape),
                    fc,
                    nn.LayerNorm(next_shape)
                )
            else:
                module = fc
            self.append_fcs.append(module)
            # set attr for pytorch to track parameters( device )
            self.__setattr__("append_fc{}".format(i), module)
            append_input_shape = next_shape

        self.last = nn.Linear(append_input_shape, output_shape)
        # print(output_shape)
        net_last_init_func(self.last)

    def forward(self, x):
        out = self.base(x)

        for append_fc in self.append_fcs:
            out = append_fc(out)
            out = self.activation_func(out)

        out = self.last(out)
        return out


class FlattenNet(Net):
    def forward(self, input):
        out = torch.cat(input, dim = -1)
        return super().forward(out)


def null_activation(x):
    return x

class ModularGatedCascadeCondNet(nn.Module):
    def __init__(self, output_shape,
            base_type, em_input_shape, input_shape,
            em_hidden_shapes,
            hidden_shapes,
            shared_base_hidden_shapes,
            num_layers, num_modules,

            module_hidden,

            gating_hidden, num_gating_layers,

            # gated_hidden
            add_bn = False,
            add_ln = False,
            dropout_neuron = 0,
            dropout_module = 0,
            pre_softmax = False,
            cond_ob = True,
            module_hidden_init_func = init.basic_init,
            last_init_func = init.uniform_init,
            activation_func = F.relu,
            shared_base = False,
             **kwargs ):

        super().__init__()

        self.base = base_type( 
            last_activation_func = null_activation,
            input_shape = input_shape,
            activation_func = activation_func,
            hidden_shapes = hidden_shapes,
            **kwargs)
        
        self.em_base = base_type(
            last_activation_func = null_activation,
            input_shape = em_input_shape,
            activation_func = activation_func,
            hidden_shapes = em_hidden_shapes,
            **kwargs )
        
        self.shared_base_flag=shared_base
        if shared_base:
            self.shared_base = base_type(
                last_activation_func = null_activation,
                input_shape = hidden_shapes[-1],
                activation_func = activation_func,
                hidden_shapes = shared_base_hidden_shapes,
                **kwargs )

        self.activation_func = activation_func
        self.dropout_neuron = dropout_neuron
        self.dropout_module = dropout_module

        module_input_shape = self.base.output_shape
        self.layer_modules = []

        self.num_layers = num_layers
        self.num_modules = num_modules

        print("add batch norm: ", add_bn)
        print("add layer norm: ", add_ln)
        for i in range(num_layers):
            layer_module = []
            for j in range( num_modules ):
                fc = nn.Linear(module_input_shape, module_hidden)
                module_hidden_init_func(fc)
                if add_bn:
                    module = nn.Sequential(
                        nn.BatchNorm1d(module_input_shape),
                        fc,
                        nn.BatchNorm1d(module_hidden)
                    )
                elif add_ln:
                    module = nn.Sequential(
                        nn.LayerNorm(module_input_shape),
                        fc,
                        nn.LayerNorm(module_hidden)
                    )
                # elif dropout>0:
                #     print("with dropout!")
                #     module = nn.Sequential(
                #         fc,
                #         nn.Dropout(dropout)
                #     )
                else:
                    module = fc

                layer_module.append(module)
                self.__setattr__("module_{}_{}".format(i,j), module)

            module_input_shape = module_hidden
            self.layer_modules.append(layer_module)

        self.last = nn.Linear(module_input_shape, output_shape)
        last_init_func( self.last )

        assert self.em_base.output_shape == self.base.output_shape, \
            "embedding should has the same dimension with base output for gated" 
        gating_input_shape = self.em_base.output_shape
        
        # 1. gating layers
        self.gating_fcs = []
        for i in range(num_gating_layers):
            gating_fc = nn.Linear(gating_input_shape, gating_hidden)
            module_hidden_init_func(gating_fc)
            self.gating_fcs.append(gating_fc)
            self.__setattr__("gating_fc_{}".format(i), gating_fc)
            gating_input_shape = gating_hidden

        # 2. gating weight layers and gating_weight_cond_fcs = n_layer - 1
        self.gating_weight_fcs = []
        self.gating_weight_cond_fcs = []

        # 3. gating weight_fc_0
        self.gating_weight_fc_0 = nn.Linear(gating_input_shape,
                    num_modules * num_modules )
        last_init_func( self.gating_weight_fc_0)

        for layer_idx in range(num_layers-2):
            gating_weight_cond_fc = nn.Linear((layer_idx+1) * \
                                               num_modules * num_modules,
                                              gating_input_shape)
            module_hidden_init_func(gating_weight_cond_fc)
            self.__setattr__("gating_weight_cond_fc_{}".format(layer_idx+1),
                             gating_weight_cond_fc)
            self.gating_weight_cond_fcs.append(gating_weight_cond_fc)

            gating_weight_fc = nn.Linear(gating_input_shape,
                                         num_modules * num_modules)
            last_init_func(gating_weight_fc)
            self.__setattr__("gating_weight_fc_{}".format(layer_idx+1),
                             gating_weight_fc)
            self.gating_weight_fcs.append(gating_weight_fc)

        # 4. gating weight cond last
        self.gating_weight_cond_last = nn.Linear((num_layers-1) * \
                                                 num_modules * num_modules,
                                                 gating_input_shape)
        module_hidden_init_func(self.gating_weight_cond_last)

        # 5. gating weight last
        self.gating_weight_last = nn.Linear(gating_input_shape, num_modules)
        last_init_func( self.gating_weight_last )

        self.pre_softmax = pre_softmax
        self.cond_ob = cond_ob

    def forward(self, x, embedding_input, return_weights = False, dropout=False):
        # Return weights for visualization
        out = self.base(x)
        embedding = self.em_base(embedding_input)

        if self.cond_ob:
            embedding = embedding * out

        out = self.activation_func(out)

        if len(self.gating_fcs) > 0:
            embedding = self.activation_func(embedding)
            for fc in self.gating_fcs[:-1]:
                embedding = fc(embedding)
                embedding = self.activation_func(embedding)
            embedding = self.gating_fcs[-1](embedding)

        base_shape = embedding.shape[:-1]

        weights = []
        flatten_weights = []
        
        # calculate weight: staring with layer 0-1 (because there must be at least 2 layers)
        raw_weight = self.gating_weight_fc_0(self.activation_func(embedding))

        weight_shape = base_shape + torch.Size([self.num_modules,
                                                self.num_modules])
        flatten_shape = base_shape + torch.Size([self.num_modules * \
                                                self.num_modules])

        raw_weight = raw_weight.view(weight_shape)

        softmax_weight = F.softmax(raw_weight, dim=-1)
        weights.append(softmax_weight)
        if self.pre_softmax:
            flatten_weights.append(raw_weight.view(flatten_shape))
        else:
            flatten_weights.append(softmax_weight.view(flatten_shape))

        for gating_weight_fc, gating_weight_cond_fc in zip(self.gating_weight_fcs, self.gating_weight_cond_fcs):
            cond = torch.cat(flatten_weights, dim=-1)
            if self.pre_softmax:
                cond = self.activation_func(cond)
            cond = gating_weight_cond_fc(cond)
            cond = cond * embedding
            cond = self.activation_func(cond)

            raw_weight = gating_weight_fc(cond)
            raw_weight = raw_weight.view(weight_shape)
            softmax_weight = F.softmax(raw_weight, dim=-1)
            weights.append(softmax_weight)
            if self.pre_softmax:
                flatten_weights.append(raw_weight.view(flatten_shape))
            else:
                flatten_weights.append(softmax_weight.view(flatten_shape))

        cond = torch.cat(flatten_weights, dim=-1)
        if self.pre_softmax:
            cond = self.activation_func(cond)
        cond = self.gating_weight_cond_last(cond)
        cond = cond * embedding
        cond = self.activation_func(cond)

        raw_last_weight = self.gating_weight_last(cond)
        last_weight = F.softmax(raw_last_weight, dim = -1)


        original_weights = []
        for w in weights:
            original_weights.append(w.detach())
            
        # dropout entire module
        if dropout:
            for i in range(len(weights)):
                # create a binary mask
                weights[i] = torch.nn.functional.dropout(weights[i], self.dropout_module, True)
        
        # go through shared base if there is any
        if self.shared_base_flag:
            out = self.shared_base(out)
        
        module_outputs = []
        for layer_module in self.layer_modules[0]:
            module_out = layer_module(out)
            # dropout neurons
            # if dropout:
                # module_out = torch.nn.functional.dropout(module_out, self.dropout_neuron, True)
            module_outputs.append(module_out.unsqueeze(-2))
        
        # module_outputs = [(layer_module(out)).unsqueeze(-2) \
        #         for layer_module in self.layer_modules[0]]
        
        module_outputs = torch.cat(module_outputs, dim = -2 )

        # [TODO] Optimize using 1 * 1 convolution.

        for i in range(self.num_layers - 1):
            new_module_outputs = []
            for j, layer_module in enumerate(self.layer_modules[i + 1]):
                module_input = (module_outputs * \
                    weights[i][..., j, :].unsqueeze(-1)).sum(dim=-2)

                module_input = self.activation_func(module_input)
                # if dropout:
                # #     new_module_outputs.append((
                # #         torch.nn.functional.dropout(layer_module(module_input), self.dropout_neuron)
                # # ).unsqueeze(-2))
                #     pass
                
                # else:
                new_module_outputs.append((
                    layer_module(module_input)
                ).unsqueeze(-2))
            
            module_outputs = torch.cat(new_module_outputs, dim = -2)

        out = (module_outputs * last_weight.unsqueeze(-1)).sum(-2)
        out = self.activation_func(out)
        out = self.last(out)
        

        if return_weights:
            return out, original_weights, last_weight
        return out


class FlattenModularGatedCascadeCondNet(ModularGatedCascadeCondNet):
    def forward(self, input, embedding_input, return_weights = False):
        out = torch.cat( input, dim = -1 )
        return super().forward(out, embedding_input, return_weights = return_weights)

 
class BootstrappedNet(Net):
    def __init__(self, output_shape, 
                 head_num = 10,
                 **kwargs ):
        self.head_num = head_num
        self.origin_output_shape = output_shape
        output_shape *= self.head_num
        super().__init__(output_shape = output_shape, **kwargs)
        
    
    def train_forward(self, x):
        '''
            get the mean and std of different heads, with grad
        '''
        base_shape = x.shape[:-1]
        # print("base shape: ", base_shape) # [batch_size, 1]
        out = super().forward(x)
        # print("out: ", out.shape)
        out_shape = base_shape + torch.Size([self.origin_output_shape, self.head_num])
        # print("out_shape:", out_shape) # [batch_size, 1, 8, 1] (8 = 4 *2) or [batch_size, 1, 1, 1]
        view_idx_shape = base_shape + torch.Size([1, 1])
        expand_idx_shape = base_shape + torch.Size([self.origin_output_shape, 1])
        # print("view idx shape: ", view_idx_shape)
        # print("expane idx shape: ", expand_idx_shape)
        
        out = out.reshape(out_shape)
        # print(out.shape)
        print(out[0])
        dist_out = []
        for i in range(out.shape[2]):
            idx = torch.LongTensor([i]).repeat(out.shape[0], 1)
            
            idx = idx.view(view_idx_shape)
            idx = idx.expand(expand_idx_shape)
            out_ = out.gather(-1, idx).squeeze(-1)
            dist_out.append(out_)
            # print("final out: ", out_.shape)
        # exit(0)
        return dist_out

    # TODO: understand this part!
    def forward(self, x, idx):
        base_shape = x.shape[:-1]
        # print("base shape: ", base_shape) # [batch_size, 1]
        out = super().forward(x)
        # print("out: ", out.shape)
        out_shape = base_shape + torch.Size([self.origin_output_shape, self.head_num])
        # print("out_shape:", out_shape) # [batch_size, 1, 8, 1] (8 = 4 *2) or [batch_size, 1, 1, 1]
        view_idx_shape = base_shape + torch.Size([1, 1])
        expand_idx_shape = base_shape + torch.Size([self.origin_output_shape, 1])
        # print("view idx shape: ", view_idx_shape)
        # print("expane idx shape: ", expand_idx_shape)
        
        out = out.reshape(out_shape)
        idx = idx.view(view_idx_shape)
        idx = idx.expand(expand_idx_shape)
        out = out.gather(-1, idx).squeeze(-1)
        return out


class FlattenBootstrappedNet(BootstrappedNet):
    def forward(self, input, idx ):
        out = torch.cat( input, dim = -1 )
        return super().forward(out, idx)
