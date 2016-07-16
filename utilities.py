import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    #interp_t_values = np.zeros_like(source,dtype=float)
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

def sitk_show(nda, title=None, margin=0.0, dpi=40):
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi

    extent = (0, nda.shape[1], nda.shape[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    plt.set_cmap("gray")
    for k in range(0,nda.shape[2]):
        print "printing slice "+str(k)
        ax.imshow(np.squeeze(nda[:,:,k]),extent=extent,interpolation=None)
        plt.draw()
        plt.pause(0.1)
        #plt.waitforbuttonpress()

def computeQualityMeasures(lP,lT):
    quality=dict()
    labelPred=sitk.GetImageFromArray(lP, isVector=False)
    labelTrue=sitk.GetImageFromArray(lT, isVector=False)
    hausdorffcomputer=sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue>0.5,labelPred>0.5)
    quality["avgHausdorff"]=hausdorffcomputer.GetAverageHausdorffDistance()
    quality["Hausdorff"]=hausdorffcomputer.GetHausdorffDistance()

    dicecomputer=sitk.LabelOverlapMeasuresImageFilter()
    dicecomputer.Execute(labelTrue>0.5,labelPred>0.5)
    quality["dice"]=dicecomputer.GetDiceCoefficient()

    return quality


def produceRandomlyDeformedImage(image, label, numcontrolpoints, stdDef):
    sitkImage=sitk.GetImageFromArray(image, isVector=False)
    sitklabel=sitk.GetImageFromArray(label, isVector=False)

    transfromDomainMeshSize=[numcontrolpoints]*sitkImage.GetDimension()

    tx = sitk.BSplineTransformInitializer(sitkImage,transfromDomainMeshSize)


    params = tx.GetParameters()

    paramsNp=np.asarray(params,dtype=float)
    paramsNp = paramsNp + np.random.randn(paramsNp.shape[0])*stdDef

    paramsNp[0:int(len(params)/3)]=0 #remove z deformations! The resolution in z is too bad

    params=tuple(paramsNp)
    tx.SetParameters(params)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitkImage)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(tx)

    resampler.SetDefaultPixelValue(0)
    outimgsitk = resampler.Execute(sitkImage)
    outlabsitk = resampler.Execute(sitklabel)

    outimg = sitk.GetArrayFromImage(outimgsitk)
    outimg = outimg.astype(dtype=np.float32)

    outlbl = sitk.GetArrayFromImage(outlabsitk)
    outlbl = (outlbl>0.5).astype(dtype=np.float32)

    return outimg,outlbl


def produceRandomlyTranslatedImage(image, label):
    sitkImage = sitk.GetImageFromArray(image, isVector=False)
    sitklabel = sitk.GetImageFromArray(label, isVector=False)

    itemindex = np.where(label > 0)
    randTrans = (0,np.random.randint(-np.min(itemindex[1])/2,(image.shape[1]-np.max(itemindex[1]))/2),np.random.randint(-np.min(itemindex[0])/2,(image.shape[0]-np.max(itemindex[0]))/2))
    translation = sitk.TranslationTransform(3, randTrans)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitkImage)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(translation)

    outimgsitk = resampler.Execute(sitkImage)
    outlabsitk = resampler.Execute(sitklabel)

    outimg = sitk.GetArrayFromImage(outimgsitk)
    outimg = outimg.astype(dtype=np.float32)

    outlbl = sitk.GetArrayFromImage(outlabsitk) > 0 # Just translation
    outlbl = outlbl.astype(dtype=np.float32)

    return outimg, outlbl

def produceRegisteredImage(image,label, fixed):
    """
    Registers an image and its label to a fixed image. Uesful for data augmentation

    Arguments:
    -----------
        image: np.ndarray
            Image to be transformed
        label: np.ndarray
            label to be transformed
        fixed: nd.ndarray
            image which the image will be registered to
    Returns:
    -----------
        final_image: np.ndarray
            The transformed output image
        final_label: np.ndarray
            The transformed output label
    """

    image=sitk.Cast(sitk.GetImageFromArray(image, isVector=False), sitk.sitkFloat32)
    label=sitk.Cast(sitk.GetImageFromArray(label, isVector=False), sitk.sitkFloat32)
    fixed=sitk.Cast(sitk.GetImageFromArray(fixed, isVector=False), sitk.sitkFloat32)
    
    initial_transform = sitk.CenteredTransformInitializer(image,fixed, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    
    image = sitk.Resample(image, fixed, initial_transform, sitk.sitkLinear, 0.0, image.GetPixelIDValue())

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)
    registration_method.SetInterpolator(sitk.sitkLinear) #2. Replace with sitkLinear
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100) 
    registration_method.SetOptimizerScalesFromPhysicalShift() 

    registration_method.SetMovingInitialTransform(initial_transform)

    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    final_transform = registration_method.Execute(sitk.Cast(image, sitk.sitkFloat32), 
                                              sitk.Cast(fixed, sitk.sitkFloat32))
    final_image= sitk.Resample(image, fixed, final_transform, sitk.sitkLinear, 0.0, image.GetPixelIDValue())
    final_label= sitk.Resample(label, fixed, final_transform, sitk.sitkLinear, 0.0, label.GetPixelIDValue())
    
    final_image = sitk.GetArrayFromImage(final_image)
    final_image = final_image.astype(dtype=np.float32)

    final_label = sitk.GetArrayFromImage(final_label)
    final_label = (final_label>0.5).astype(dtype=np.float32)

    return final_image, final_label