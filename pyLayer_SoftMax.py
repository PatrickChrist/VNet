import caffe
import numpy as np
import utils

class DiceLoss(caffe.Layer):
    """
    Compute energy based on dice coefficient.
    """
    result = None
    gt = None
    weightmap=None

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute the dice. the result of the softmax and the ground truth.")



    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != 2*bottom[1].count:
            print bottom[0].data.shape
            print bottom[1].data.shape
            raise Exception("the dimension of inputs should match")

        # loss output is two scalars (mean and std)
        top[0].reshape(1)

    def forward(self, bottom, top):
        # Load Softmaxed data blobs
        self.result = np.squeeze(bottom[0].data[...])
        self.gt = np.squeeze(bottom[1].data[...])
        self.weightmap=np.zeros_like(bottom[1].data[...])
        
        # Initalize the data loss
        data_loss = 0
        # Iterate over batch-size
        for i in range(0,bottom[0].data.shape[0]):
            probs = self.result[i,:,:] # Batchsize i
            gt = self.gt[i,:].astype(np.int8)    
	    weightmap=utils.generate_weightmap_from_label(gt,class_weight=1000000,border_thickness=2,stepness=0.1,shape=(128,128,64))
            self.weightmap[i,:]=weightmap
            # Get the Weighting Factor
            weightfac=np.sum(weightmap.ravel())     
            correct_logprobs = np.multiply(-np.log(probs[gt,range(len(gt))]),weightmap)
            data_loss =data_loss+np.sum(correct_logprobs)/weightfac


        top[0].data[...] = data_loss
         # Use this to debug backprob
         # from IPython.core.debugger import Tracer
         # Tracer()() #this one triggers the debugger

    def backward(self, top, propagate_down, bottom):
        for btm in [0]:
            for i in range(0, bottom[btm].diff.shape[0]):
                probs = self.result[i,:,:] # Batchsize i
                
                print "Probs shape is {}" .format(probs.shape)
                
                gt = self.gt[i,:].astype(np.int8)
                
                print "GT shape is {}" .format(gt.shape)
                
                weightfac=np.sum(self.weightmap[i,:])
                
                print "Bottom shape is {}" .format(bottom[btm].diff[i, :, :].shape)
                print "Weightmap shape is {}" .format(self.weightmap[i,:].shape)
                
                bottom[btm].diff[i, :, :] = probs
                bottom[btm].diff[i, :, :][gt,range(len(gt))] -= 1
                # Multipy elementwise prob with weightmap
                for label in range(bottom[btm].diff.shape[1]):
                    bottom[btm].diff[i, label, :] =np.multiply(bottom[btm].diff[i, label, :],self.weightmap[i,:])
                    bottom[btm].diff[i, label, :] /= weightfac
                
                # Use this to debug backprob
                #from IPython.core.debugger import Tracer
                #Tracer()() #this one triggers the debugger
