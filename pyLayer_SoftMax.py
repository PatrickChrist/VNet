import caffe
import numpy as np

class DiceLoss(caffe.Layer):
    """
    Compute energy based on dice coefficient.
    """
    union = None
    intersection = None
    result = None
    gt = None

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
        
        # Initalize the data loss
        data_loss = 0
        # Iterate over batch-size
        for i in range(0,bottom[0].data.shape[0]):
            probs = self.result[i,:,:] # Batchsize i
            gt = self.gt[i,:].astype(np.int8)
            lenarray=len(gt)
            correct_logprobs = -np.log(probs[gt,range(lenarray)])
            data_loss =data_loss+np.sum(correct_logprobs)/len(gt)


        top[0].data[...] = data_loss

    def backward(self, top, propagate_down, bottom):
        for btm in [0]:
            for i in range(0, bottom[btm].diff.shape[0]):
                probs = self.result[i,:,:] # Batchsize i
                gt = self.gt[i,:].astype(np.int8)
                lenarray=len(gt)
                bottom[btm].diff[i, :, :] = probs
                bottom[btm].diff[i, :, :][gt,range(lenarray)] -= 1
                bottom[btm].diff[i, :, :] /= lenarray
