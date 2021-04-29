# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 13:25:35 2018

reference: https://github.com/BertMoons/QuantizedNeuralNetworks-Keras-Tensorflow
all the credit refer to BertMoons on QuantizedNeuralNetworks-Keras-Tensorflow

@author: Yung-Yu Tsai

"""

import h5py, os
import numpy as np
from scipy import stats


def load_attributes_from_hdf5_group(group, name):
    """Loads attributes of the specified name from the HDF5 group.

    This method deals with an inherent problem
    of HDF5 file which is not able to store
    data larger than HDF5_OBJECT_HEADER_LIMIT bytes.

    Arguments
    ---------
    group: A pointer to a HDF5 group.
    name: A name of the attributes to load.

    Returns
    -------
    data: Attributes data.
    """
    if name in group.attrs:
        data = [n.decode('utf8') for n in group.attrs[name]]
    else:
        data = []
        chunk_id = 0
        while ('%s%d' % (name, chunk_id)) in group.attrs:
            data.extend([n.decode('utf8')
                         for n in group.attrs['%s%d' % (name, chunk_id)]])
            chunk_id += 1
    return data
    

def convert_original_weight_layer_name(original_weight_name,quantized_weight_name=None,print_detail=False):
        
    if quantized_weight_name is None:
        quantized_weight_name=original_weight_name[:-3]+'_quantized.h5'
        if os.path.isfile(quantized_weight_name):
            if print_detail:
                print('quantized layer name weight already exist skip conversion and continue...')
            return quantized_weight_name
        else:
            q_weight_f = h5py.File(quantized_weight_name,'w')
    else:
        if os.path.isfile(quantized_weight_name):
            if print_detail:
                print('quantized layer name weight already exist skip conversion and continue...')
            return quantized_weight_name
        else:
            q_weight_f = h5py.File(quantized_weight_name,'w')
    
    
    o_weight_f = h5py.File(original_weight_name,'r')
        
    if 'keras_version' in o_weight_f.attrs:
            original_keras_version = o_weight_f.attrs['keras_version'].decode('utf8')
    else:
        original_keras_version = '1'
    if 'backend' in o_weight_f.attrs:
        original_backend = o_weight_f.attrs['backend'].decode('utf8')
    else:
        original_backend = 'None'
        
    
    layer_names = load_attributes_from_hdf5_group(o_weight_f, 'layer_names')
    filtered_layer_names = []
    for layer_name in layer_names:
        o_group = o_weight_f[layer_name]
        weight_names = load_attributes_from_hdf5_group(o_group, 'weight_names')
        if weight_names:
            filtered_layer_names.append(layer_name)
            
    quantized_layer_names = []
    for layer in layer_names:
        if layer in filtered_layer_names:
            quantized_layer_names.append('quantized_'+layer)
        else:
            quantized_layer_names.append(layer)
            
    q_weight_f.attrs.create('layer_names',[temp.encode('utf8') for temp in quantized_layer_names])
    q_weight_f.attrs.create('backend',original_backend.encode('utf8'))
    q_weight_f.attrs.create('keras_version',original_keras_version.encode('utf8'))
    
    for layer_iter, layer_name in enumerate(layer_names):
        o_group = o_weight_f[layer_name]
        weight_names = load_attributes_from_hdf5_group(o_group, 'weight_names')
        weight_values = [np.asarray(o_group[weight_name]) for weight_name in weight_names]
        quantized_layer = q_weight_f.create_group(quantized_layer_names[layer_iter])
        if layer_name in filtered_layer_names:
            quantized_layer.attrs.create('weight_names',[('quantized_'+temp).encode('utf8') for temp in weight_names])
        else:
            quantized_layer.attrs.create('weight_names',[temp.encode('utf8') for temp in weight_names])
        if len(weight_names) != 0:
            quantized_sublayer = quantized_layer.create_group(quantized_layer_names[layer_iter])
        
        for weight_iter, weight_name in enumerate(weight_names):
            quantized_sublayer.create_dataset(weight_name[len(layer_name)+1:],weight_values[weight_iter].shape,weight_values[weight_iter].dtype,weight_values[weight_iter])
        
        
    o_weight_f.close()
    q_weight_f.close()
    
    return quantized_weight_name


def quantize_weight(original_weight_name, weight_bit_width, weight_factorial_bit, quantized_weight_name=None, rounding_method='nearest'):
    o_weight_f = h5py.File(original_weight_name,'r')
    if quantized_weight_name is None:
        quantized_weight_name=original_weight_name[:-3]+('_quantized_%s_rounding_%dB%dI%dF.h5' % (rounding_method,weight_bit_width, weight_bit_width-weight_factorial_bit-1, weight_factorial_bit))
        q_weight_f = h5py.File(quantized_weight_name,'w')
    else:
        q_weight_f = h5py.File(quantized_weight_name,'w')
        
        
    if 'keras_version' in o_weight_f.attrs:
            original_keras_version = o_weight_f.attrs['keras_version'].decode('utf8')
    else:
        original_keras_version = '1'
    if 'backend' in o_weight_f.attrs:
        original_backend = o_weight_f.attrs['backend'].decode('utf8')
    else:
        original_backend = None
        
    
    layer_names = load_attributes_from_hdf5_group(o_weight_f, 'layer_names')
    filtered_layer_names = []
    for layer_name in layer_names:
        o_group = o_weight_f[layer_name]
        weight_names = load_attributes_from_hdf5_group(o_group, 'weight_names')
        if weight_names:
            filtered_layer_names.append(layer_name)
            
            
    q_weight_f.attrs.create('layer_names',[temp.encode('utf8') for temp in layer_names])
    q_weight_f.attrs.create('backend',original_backend.encode('utf8'))
    q_weight_f.attrs.create('keras_version',original_keras_version.encode('utf8'))
    
    for layer_iter, layer_name in enumerate(layer_names):
        o_group = o_weight_f[layer_name]
        weight_names = load_attributes_from_hdf5_group(o_group, 'weight_names')
        weight_values = [np.asarray(o_group[weight_name]) for weight_name in weight_names]
        quantized_layer = q_weight_f.create_group(layer_names[layer_iter])
        quantized_layer.attrs.create('weight_names',[temp.encode('utf8') for temp in weight_names])
        if len(weight_names) != 0:
            quantized_sublayer = quantized_layer.create_group(layer_names[layer_iter])
        
        for weight_iter, weight_name in enumerate(weight_names):
            m = np.power(2,weight_factorial_bit)
            quantized_weight_value = weight_values[weight_iter] * m
            
            if rounding_method == 'nearest':
                quantized_weight_value = np.round(quantized_weight_value)
            elif rounding_method == 'zero':
                quantized_weight_value = np.trunc(quantized_weight_value)
            elif rounding_method == 'down':
                quantized_weight_value = np.floor(quantized_weight_value)
            elif rounding_method == 'stochastic':
                if np.average(quantized_weight_value-np.floor(quantized_weight_value)) > 0.5:
                    quantized_weight_value = np.ceil(quantized_weight_value)
                else:
                    quantized_weight_value = np.floor(quantized_weight_value)
            else:
                print('Wrong Rounding Type\nChoose between \'nearest\' , \'zero\' , \'down\'')
                
            quantized_weight_value = np.clip(quantized_weight_value/m, -np.power(2,weight_bit_width-weight_factorial_bit-1), np.power(2,weight_bit_width-weight_factorial_bit-1)-np.power(0.5,weight_factorial_bit))
                
            quantized_sublayer.create_dataset(weight_name[len(layer_name)+1:],weight_values[weight_iter].shape,weight_values[weight_iter].dtype,quantized_weight_value)
        
        
    o_weight_f.close()
    q_weight_f.close()
    
    return quantized_weight_name

def fuse_BN_weight(original_weight_name,fused_weight_name=None,epsilon=1e-7):
    
    #================================================================
    possible_BN_name_list = ['bn','BN','BatchNormalization','batch']
    #================================================================


    o_weight_f = h5py.File(original_weight_name,'r')
    if fused_weight_name is None:
        fused_weight_name=original_weight_name[:-3]+('_fused_BN.h5')
        if os.path.isfile(fused_weight_name):
            print('Fused BN weight already exist skip conversion and continue...')
            return fused_weight_name
        else:
            f_weight_f = h5py.File(fused_weight_name,'w')
    else:
        if os.path.isfile(fused_weight_name):
            print('Fused BN weight already exist skip conversion and continue...')
            return fused_weight_name
        else:
            f_weight_f = h5py.File(fused_weight_name,'w')
        
        
    if 'keras_version' in o_weight_f.attrs:
        original_keras_version = o_weight_f.attrs['keras_version'].decode('utf8')
    else:
        original_keras_version = '1'
    if 'backend' in o_weight_f.attrs:
        original_backend = o_weight_f.attrs['backend'].decode('utf8')
    else:
        original_backend = 'None'
        
    
    layer_names = load_attributes_from_hdf5_group(o_weight_f, 'layer_names')
    fused_layer_names = []
    for layer_iter,layer_name in enumerate(layer_names[:-1]):
        pred_list=[(keyword in layer_names[layer_iter+1]) for keyword in possible_BN_name_list]
        if any(pred_list):
            fused_layer_names.append(layer_name)
        
    new_layer_names = []
    bn_layer_names = []
    for layer_name in layer_names:
        pred_list=[(keyword in layer_name) for keyword in possible_BN_name_list]
        if any(pred_list):
            bn_layer_names.append(layer_name)
        else:
            new_layer_names.append(layer_name)

            
            
    f_weight_f.attrs.create('layer_names',[temp.encode('utf8') for temp in new_layer_names])
    f_weight_f.attrs.create('backend',original_backend.encode('utf8'))
    f_weight_f.attrs.create('keras_version',original_keras_version.encode('utf8'))
    
    for layer_iter, layer_name in enumerate(layer_names):
        if layer_name not in bn_layer_names:
            o_group = o_weight_f[layer_name]
            weight_names = load_attributes_from_hdf5_group(o_group, 'weight_names')
            weight_values = [np.asarray(o_group[weight_name]) for weight_name in weight_names]
            fused_layer = f_weight_f.create_group(layer_names[layer_iter])
            fused_layer.attrs.create('weight_names',[temp.encode('utf8') for temp in weight_names])
            if len(weight_names) != 0:
                fused_sublayer = fused_layer.create_group(layer_names[layer_iter])
            
            if layer_name in fused_layer_names:
                bn_o_group = o_weight_f[layer_names[layer_iter+1]]
                bn_weight_names = load_attributes_from_hdf5_group(bn_o_group, 'weight_names')
                bn_weight_values = [np.asarray(bn_o_group[weight_name]) for weight_name in bn_weight_names]
                
                if weight_values[0].shape[-1]==len(bn_weight_values[0]):
                    # normal circumstance output channel at last dimension
                    shape_tmp1 = []
                    shape_tmp2 = []
                    for i in range(len(weight_values[0].shape)-1):
                        shape_tmp1.append(1)
                        shape_tmp2.append(weight_values[0].shape[i])
                    shape_tmp1.append(len(bn_weight_values[0]))
                    shape_tmp2.append(1)
                elif weight_values[0].shape[-2]==len(bn_weight_values[0]) and weight_values[0].shape[-1]==1:
                    # depthwise convolution 
                    shape_tmp1 = []
                    shape_tmp2 = []
                    for i in range(len(weight_values[0].shape)-2):
                        shape_tmp1.append(1)
                        shape_tmp2.append(weight_values[0].shape[i])
                    shape_tmp1.append(len(bn_weight_values[0]))
                    shape_tmp1.append(1)
                    shape_tmp2.append(1)
                    shape_tmp2.append(1)
                    
                gamma_tmp = bn_weight_values[0]
                beta_tmp = bn_weight_values[1]
                mean_tmp = bn_weight_values[2]
                variance_tmp = bn_weight_values[3]
                
                coef = np.divide(gamma_tmp,np.sqrt(variance_tmp+epsilon))
                const = beta_tmp-np.multiply(mean_tmp,coef)
                
                # for those layer without bias
                if len(weight_names)==1:
                    fused_bias = const
                else:
                    fused_bias = np.multiply(coef,weight_values[1])+const
                
                coef = np.tile(np.reshape(coef,shape_tmp1),shape_tmp2)
                
                fused_kernel = np.multiply(weight_values[0],coef)
                
                    
                fused_sublayer.create_dataset(weight_names[0][len(layer_name)+1:],weight_values[0].shape,weight_values[0].dtype,fused_kernel)
                # for those layer without bias
                if len(weight_names)==1:
                    fused_layer.attrs.create('weight_names',[weight_names[0].encode('utf8'),weight_names[0].replace('kernel','bias').encode('utf8')])
                    fused_sublayer.create_dataset(weight_names[0][len(layer_name)+1:].replace('kernel','bias'),bn_weight_values[0].shape,bn_weight_values[0].dtype,fused_bias)
                else:
                    fused_sublayer.create_dataset(weight_names[1][len(layer_name)+1:],weight_values[1].shape,weight_values[1].dtype,fused_bias)

                        
            else:
                for weight_iter, weight_name in enumerate(weight_names):
                    fused_sublayer.create_dataset(weight_name[len(layer_name)+1:],weight_values[weight_iter].shape,weight_values[weight_iter].dtype,weight_values[weight_iter])
        
        
    o_weight_f.close()
    f_weight_f.close()
    
    return fused_weight_name


# for the unusual resnet50 weight file
def fuse_BN_weight_beta(original_weight_name,layer_names=None,fused_weight_name=None,epsilon = 1e-7):
    
    #================================================================
    possible_BN_name_list = ['bn','BN','BatchNormalization','batch']
    #================================================================


    o_weight_f = h5py.File(original_weight_name,'r')
    if fused_weight_name is None:
        fused_weight_name=original_weight_name[:-3]+('_fused_BN.h5')
        if os.path.isfile(fused_weight_name):
            print('Fused BN weight already exist skip conversion and continue...')
            return fused_weight_name
        else:
            f_weight_f = h5py.File(fused_weight_name,'w')
    else:
        if os.path.isfile(fused_weight_name):
            print('Fused BN weight already exist skip conversion and continue...')
            return fused_weight_name
        else:
            f_weight_f = h5py.File(fused_weight_name,'w')
        
        
    if 'keras_version' in o_weight_f.attrs:
        original_keras_version = o_weight_f.attrs['keras_version'].decode('utf8')
    else:
        original_keras_version = '1'
    if 'backend' in o_weight_f.attrs:
        original_backend = o_weight_f.attrs['backend'].decode('utf8')
    else:
        original_backend = 'None'
        
    if layer_names is None:
        layer_names = load_attributes_from_hdf5_group(o_weight_f, 'layer_names')
    fused_layer_names = []
    for layer_iter,layer_name in enumerate(layer_names[:-1]):
        pred_list=[(keyword in layer_names[layer_iter+1]) for keyword in possible_BN_name_list]
        if any(pred_list):
            fused_layer_names.append(layer_name)
        
    new_layer_names = []
    bn_layer_names = []
    for layer_name in layer_names:
        pred_list=[(keyword in layer_name) for keyword in possible_BN_name_list]
        if any(pred_list):
            bn_layer_names.append(layer_name)
        else:
            new_layer_names.append(layer_name)

            
            
    f_weight_f.attrs.create('layer_names',[temp.encode('utf8') for temp in new_layer_names])
    f_weight_f.attrs.create('backend',original_backend.encode('utf8'))
    f_weight_f.attrs.create('keras_version',original_keras_version.encode('utf8'))
    
    for layer_iter, layer_name in enumerate(layer_names):
        if layer_name not in bn_layer_names:
            o_group = o_weight_f[layer_name]
            weight_names = load_attributes_from_hdf5_group(o_group, 'weight_names')
            weight_values = [np.asarray(o_group[weight_name]) for weight_name in weight_names]
            fused_layer = f_weight_f.create_group(layer_names[layer_iter])
            fused_layer.attrs.create('weight_names',[temp.encode('utf8') for temp in weight_names])
            if len(weight_names) != 0:
                fused_sublayer = fused_layer#.create_group(layer_names[layer_iter])
            
            if layer_name in fused_layer_names:
                bn_o_group = o_weight_f[layer_names[layer_iter+1]]
                bn_weight_names = load_attributes_from_hdf5_group(bn_o_group, 'weight_names')
                bn_weight_values = [np.asarray(bn_o_group[weight_name]) for weight_name in bn_weight_names]
                
                if weight_values[0].shape[-1]==len(bn_weight_values[0]):
                    # normal circumstance output channel at last dimension
                    shape_tmp1 = []
                    shape_tmp2 = []
                    for i in range(len(weight_values[0].shape)-1):
                        shape_tmp1.append(1)
                        shape_tmp2.append(weight_values[0].shape[i])
                    shape_tmp1.append(len(bn_weight_values[0]))
                    shape_tmp2.append(1)
                elif weight_values[0].shape[-2]==len(bn_weight_values[0]) and weight_values[0].shape[-1]==1:
                    # depthwise convolution 
                    shape_tmp1 = []
                    shape_tmp2 = []
                    for i in range(len(weight_values[0].shape)-2):
                        shape_tmp1.append(1)
                        shape_tmp2.append(weight_values[0].shape[i])
                    shape_tmp1.append(len(bn_weight_values[0]))
                    shape_tmp1.append(1)
                    shape_tmp2.append(1)
                    shape_tmp2.append(1)
                    
                gamma_tmp = bn_weight_values[0]
                beta_tmp = bn_weight_values[1]
                mean_tmp = bn_weight_values[2]
                variance_tmp = bn_weight_values[3]
                
                coef = np.divide(gamma_tmp,np.sqrt(variance_tmp+epsilon))
                const = beta_tmp-np.multiply(mean_tmp,coef)
                
                # for those layer without bias
                if len(weight_names)==1:
                    fused_bias = const
                else:
                    fused_bias = np.multiply(coef,weight_values[1])+const
                
                coef = np.tile(np.reshape(coef,shape_tmp1),shape_tmp2)
                
                fused_kernel = np.multiply(weight_values[0],coef)
                
                    
                fused_sublayer.create_dataset(weight_names[0],weight_values[0].shape,weight_values[0].dtype,fused_kernel)
                # for those layer without bias
                if len(weight_names)==1:
                    fused_layer.attrs.create('weight_names',[weight_names[0].encode('utf8'),weight_names[0].replace('kernel','bias').encode('utf8')])
                    fused_sublayer.create_dataset(weight_names[0].replace('kernel','bias'),bn_weight_values[0].shape,bn_weight_values[0].dtype,fused_bias)
                else:
                    fused_sublayer.create_dataset(weight_names[1],weight_values[1].shape,weight_values[1].dtype,fused_bias)

                        
            else:
                for weight_iter, weight_name in enumerate(weight_names):
                    fused_sublayer.create_dataset(weight_name,weight_values[weight_iter].shape,weight_values[weight_iter].dtype,weight_values[weight_iter])
        
        
    o_weight_f.close()
    f_weight_f.close()
    
    return fused_weight_name

def view_weight_distribution(weight_name, quantiles=None, bins=None):
    """ View the weight distribution and return the statistics into information list

    Parameters
    ----------
    weight_name : String
        The weight file name for distribution check.
    quantiles: Float or List of Float. 
        | The quantiles for layer weight distribution inpection.
        | Default [0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]
    bins: Integer
        The number of bins for layer weight histogram inspection.

    Returns
    -------
    model_weight_distribution: List of List of Dictionary
        The weight distribution information for given model. Outer List index is the same as layer index in model.
        Inner List stands for [kernel, bias] and each element contains statistic information.
        
        >>> [[{'mean':average,'std_dev':standard_deviation,'kstest':KstestResult}, #L1 kernal
        ...   {'mean':average,'std_dev':standard_deviation,'kstest':KstestResult}], #L1 bias
        ...  [{'mean':average,'std_dev':standard_deviation,'kstest':KstestResult}, #L2 kernal
        ...   {'mean':average,'std_dev':standard_deviation,'kstest':KstestResult}], #L2 bias
        ...  --- ]

    """
    o_weight_f = h5py.File(weight_name,'r')        
        
    if quantiles is None:
        quantiles=np.linspace(0,1,11)
    if bins is None:
        bins=100
    
    layer_names = load_attributes_from_hdf5_group(o_weight_f, 'layer_names')
    model_weight_distribution = [None for i in range(len(layer_names))]
                
    for layer_iter, layer_name in enumerate(layer_names):
        o_group = o_weight_f[layer_name]
        weight_names = load_attributes_from_hdf5_group(o_group, 'weight_names')
        weight_values = [np.asarray(o_group[weight_name]) for weight_name in weight_names]
        if len(weight_names) != 0:        
            layer_distinfo=list()
            for weight_iter, weight_name in enumerate(weight_names):
                analyze_wght = weight_values[weight_iter]
                analyze_wght=analyze_wght.flatten()
                
                mean,stddev=stats.norm.fit(analyze_wght)
                diststat=stats.norm(loc=mean,scale=stddev)
                ksresult=list(stats.kstest(analyze_wght.flatten(),diststat.cdf))
                quantile_values=np.quantile(analyze_wght,quantiles)
                hist,bin_edges=np.histogram(analyze_wght, bins=bins)
                
                dist_info=dict()
                dist_info['weight_name']=weight_name
                dist_info['mean']=mean
                dist_info['std_dev']=stddev
                dist_info['kstest']=ksresult
                dist_info['quantile']=quantile_values.astype(np.float32)
                dist_info['hist']=hist.astype(np.int32)
                dist_info['bin_edges']=bin_edges
                
                layer_distinfo.append(dist_info)
            
            model_weight_distribution[layer_iter]=layer_distinfo
        
        
    o_weight_f.close()
    model_weight_distribution.insert(0,None)
    
    return model_weight_distribution

